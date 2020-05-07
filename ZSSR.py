import matplotlib.pyplot as plt
from configs import Config
import matplotlib.image as img
import torch.nn as nn
from tqdm import tqdm
from utils import *
from net import ZSSRNet, Downnet
from torch.nn import init


class ZSSR:
    # Basic current state variables initialization / declaration
    kernel = None
    learning_rate = None
    hr_father = None
    lr_son = None
    sr = None
    sf = None
    final_sr = None
    hr_fathers_sources = []

    # Output variables initialization / declaration
    reconstruct_output = None
    train_output = None
    output_shape = None

    # Counters and logs initialization
    iter = 0
    base_sf = 1.0
    base_ind = 0
    sf_ind = 0
    mse = []
    mse_rec = []
    interp_rec_mse = []
    interp_mse = []
    mse_steps = []
    loss = []
    learning_rate_change_iter_nums = []
    fig = None

    # 构造函数
    def __init__(self, input_img, conf=Config(), ground_truth=None, kernels=None):
        self.conf = conf
        self.phases = int(math.log(self.conf.scale, 2))
        self.input = img.imread(input_img)
        self.kernels = preprocess_kernels(kernels, conf)
        self.hr_fathers_sources = [self.input]
        self.file_name = input_img if type(input_img) is str else conf.name
        ##################################net##############################
        self.device = torch.device('cpu' if conf.cpu else 'cuda')
        self.model = ZSSRNet(input_channels=3).to(self.device)
        self.downnet = Downnet().to(self.device)

    # 入口,整个run函数是在不同尺度下和kernel下的大循环
    def run(self):
        # Run gradually on all scale factors (if only one jump then this loop only happens once)
        for self.sf_ind, (sf, self.kernel) in enumerate(zip(self.conf.scale_factors, self.kernels)):
            # verbose
            print('** Start training for base sf=', sf, ' **')
            # Relative_sf (used when base change is enabled. this is when input is the output of some previous scale)
            if np.isscalar(sf):
                sf = [sf, sf]
            # sf 默认为[[2.0, 2.0]]
            self.sf = np.array(sf) / np.array(self.base_sf)
            # self.sf [2. 2.]
            self.output_shape = np.uint(np.ceil(np.array(self.input.shape[0:2]) * sf))
            self.model.apply(self.weights_init_kaiming)
            self.downnet.apply(self.weights_init_kaiming)
            # Initialize network
            self.loss = [None] * self.conf.max_iters
            self.mse, self.mse_rec, self.interp_mse, self.interp_rec_mse, self.mse_steps = [], [], [], [], []
            self.iter = 0
            self.learning_rate = self.conf.learning_rate
            self.learning_rate_change_iter_nums = [0]
            self.train()  # 训练好几轮，每次都增强,结果保存在 self.train_output
            post_processed_output = self.final_test(self.input)  # 用的是最开始的输入,结果放入hr_fathers_sources
            for i in range(self.phases - 1):
                post_processed_output = self.final_test(post_processed_output)
            self.hr_fathers_sources.append(post_processed_output)
            self.base_change()
            if self.conf.save_results:
                plt.imsave('%s/%sHR.png' %
                           (self.conf.result_path, os.path.basename(self.file_name)[:-6]),
                           post_processed_output, vmin=0, vmax=1)

            # verbose
            print('** Done training')

        # Return the final post processed output.
        # noinspection PyUnboundLocalVariable
        return post_processed_output

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('Linear') != -1:
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, 0.02)
            init.constant(m.bias.data, 0.0)

    def father_to_son(self, hr_father):
        lr_son = imresize(hr_father, 1.0 / self.sf, kernel=self.kernel)
        return np.clip(lr_son + np.random.randn(*lr_son.shape) * self.conf.noise_std, 0, 1)

    # 在每次迭代前都进行数据增强
    def train(self):
        self.model.train()
        self.downnet.train()
        # vgg = vgg16(pretrained=True)
        # loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        # loss_network.cuda()
        # for param in loss_network.parameters():
        #     param.requires_grad = False
        # self.loss_network = loss_network
        loss = nn.L1Loss()
        model_optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        downnet_optimizer = optim.Adam(self.downnet.parameters(), lr=self.learning_rate)
        # sampler 数据增强得到HR-LR pair
        self.hr_fathers_sources = [self.input]  # input 格式
        progress = tqdm(range(self.conf.max_iters))
        for self.iter in progress:  # conf.max_iters
            self.model.zero_grad()
            self.downnet.zero_grad()
            self.hr_father = random_augment(ims=self.hr_fathers_sources,
                                            base_scales=[1.0] + self.conf.scale_factors,
                                            leave_as_is_probability=self.conf.augment_leave_as_is_probability,
                                            no_interpolate_probability=self.conf.augment_no_interpolate_probability,
                                            min_scale=self.conf.augment_min_scale,
                                            max_scale=([1.0] + self.conf.scale_factors)[
                                                len(self.hr_fathers_sources) - 1],
                                            allow_rotation=self.conf.augment_allow_rotation,
                                            scale_diff_sigma=self.conf.augment_scale_diff_sigma,
                                            shear_sigma=self.conf.augment_shear_sigma,
                                            crop_size=self.conf.crop_size)

            lr_son = self.father_to_son(self.hr_father)
            lrs = []
            lr_son = self.hr_father
            lrs.append(lr_son)

            x2 = []
            x4 = []
            # 低分图像真值 和插值输入
            for i in range(int(math.log(self.conf.scale, 2))):
                lr_son = self.father_to_son(lr_son)
                lrs.insert(0, lr_son)
            # 上采样
            for i in range(len(lrs) - 1):
                # print('lr', lrs[i].shape)
                interpolated = imresize(lrs[i], None, lrs[i + 1].shape, self.conf.upscale_method)
                lrx2 = self.model(torch.tensor(interpolated, dtype=torch.float32).cuda())
                x2.append(lrx2)
                if self.phases == 2 and i < len(lrs) - 2:
                    lrx2 = lrx2.cpu().detach().numpy()
                    interpolated = imresize(lrx2, None, lrs[i + 2].shape, self.conf.upscale_method)
                    lrx4 = self.model(torch.tensor(interpolated, dtype=torch.float32).cuda())
                    x4.append(lrx4)

            interpolated = imresize(self.hr_father, self.sf, None, self.conf.upscale_method)
            target2x = self.model(torch.tensor(interpolated, dtype=torch.float32).cuda()).cpu().detach().numpy()
            interpolated = imresize(target2x, self.sf, None, self.conf.upscale_method)
            x4.append(self.model(torch.tensor(interpolated, dtype=torch.float32).cuda()))
            # print('father:', self.hr_father.shape)
            # print('lr:')
            # for i in range(len(lrs)):
            #     print(lrs[i].shape)
            # print('2x:')
            # for i in range(len(x2)):
            #     print(x2[i].shape)
            # print('4x:')
            # for i in range(len(x4)):
            #     print(x4[i].shape)

            # 转tensor
            for i in range(len(lrs)):
                lrs[i] = torch.tensor(lrs[i], dtype=torch.float32).cuda()

            # primary loss:
            loss_primary = torch.zeros(1).cuda()
            for i in range(len(lrs) - 1):
                loss_primary += loss(x2[i], lrs[i + 1])
            if self.phases == 2:
                for i in range(len(lrs) - 2):
                    loss_primary += loss(x4[i], lrs[i + 2])

            # dual loss
            target = None
            interpolated = imresize(lrs[-1].cpu().detach().numpy(), 1 / self.sf, None, self.conf.upscale_method)
            for i in range(self.phases):
                target = self.downnet(torch.tensor(interpolated, dtype=torch.float32).cuda())
                interpolated = imresize(target.cpu().detach().numpy(), 1 / self.sf, None, self.conf.upscale_method)
            loss_dual = loss(target, lrs[0])

            # circle loss
            loss_cycle = torch.zeros(1).cuda()
            if self.phases == 1:
                interpolated = imresize(x2[-1].cpu().detach().numpy(), 1 / self.sf, None, self.conf.upscale_method)
                target = self.downnet(torch.tensor(interpolated, dtype=torch.float32).cuda())
                loss_cycle += loss(target, lrs[-1])
            if self.phases == 2:
                interpolated = imresize(x4[-1].cpu().detach().numpy(), 1 / self.sf, None, self.conf.upscale_method)
                target = self.downnet(torch.tensor(interpolated, dtype=torch.float32).cuda())
                interpolated = imresize(target.cpu().detach().numpy(), 1 / self.sf, None, self.conf.upscale_method)
                target = self.downnet(torch.tensor(interpolated, dtype=torch.float32).cuda())
                loss_cycle += loss(target, lrs[-1])
            # compute total loss
            self.loss[
                self.iter] = loss_primary + self.conf.dual_weight * loss_dual + self.conf.cycle_weight * loss_cycle  # + 0.006 * perc_loss
            cpu_loss = self.loss[self.iter].data.cpu().detach().numpy()
            progress.set_description(
                "Iter: {iter} Loss: {loss}, Lr: {lr}".format(iter=self.iter, loss=cpu_loss.round(4),
                                                             lr=self.learning_rate))
            if self.iter > 0 and self.iter % 2000 == 0:
                self.learning_rate = self.learning_rate / 10
                for param_group in model_optimizer.param_groups:
                    param_group['lr'] = self.learning_rate
                for param_group in downnet_optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

            self.loss[self.iter].backward()
            model_optimizer.step()
            downnet_optimizer.step()

            # stop when minimum learning rate was passed
            if self.learning_rate < self.conf.min_learning_rate:
                print('learning_rate terminated')
                break

    def test(self, input):
        interpolated_lr_son = imresize(input, self.sf, None, self.conf.upscale_method)
        interpolated_lr_son = torch.tensor(np.ascontiguousarray(interpolated_lr_son), dtype=torch.float32).cuda()
        self.model.eval()
        sr = self.model(interpolated_lr_son).cpu()
        sr = quantize(sr, self.conf.rgb_range)
        sr = sr.detach().numpy()
        return sr

    def final_test(self, input):  # input作为测试输入,八种几何变换
        # geometric self ensemble
        outputs = []
        for k in range(0, 1 + 7 * self.conf.output_flip, 1 + int(self.sf[0] != self.sf[1])):
            # Rotate 90*k degrees and mirror flip when k>=4
            test_input = np.rot90(input, k) if k < 4 else np.fliplr(np.rot90(input, k))
            # Apply network on the rotated input,此时不计算损失回传(相当于测试)
            tmp_output = self.test(test_input)
            # Undo the rotation for the processed output (mind the opposite order of the flip and the rotation)
            tmp_output = np.rot90(tmp_output, -k) if k < 4 else np.rot90(np.fliplr(tmp_output), -k)
            # fix SR output with back projection technique for each augmentation
            for bp_iter in range(self.conf.back_projection_iters[self.sf_ind]):
                tmp_output = back_projection(tmp_output, input, down_kernel=self.kernel,
                                             up_kernel=self.conf.upscale_method)
            # save outputs from all augmentations
            outputs.append(tmp_output)

        # Take the median over all 8 outputs
        almost_final_sr = np.median(outputs, 0)

        # Again back projection for the final fused result
        for bp_iter in range(self.conf.back_projection_iters[self.sf_ind]):
            almost_final_sr = back_projection(almost_final_sr, input, down_kernel=self.kernel,
                                              up_kernel=self.conf.upscale_method)
        self.final_sr = almost_final_sr
        return self.final_sr

    def base_change(self):
        if len(self.conf.base_change_sfs) < self.base_ind + 1:
            return

        # Change base input image if required (this means current output becomes the new input)
        if abs(self.conf.scale_factors[self.sf_ind] - self.conf.base_change_sfs[self.base_ind]) < 0.001:
            if len(self.conf.base_change_sfs) > self.base_ind:
                # The new input is the current output
                self.input = self.final_sr
                # The new base scale_factor
                self.base_sf = self.conf.base_change_sfs[self.base_ind]
                # Keeping track- this is the index inside the base scales list (provided in the config)
                self.base_ind += 1
            print('base changed to %.2f' % self.base_sf)
