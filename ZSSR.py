import matplotlib.pyplot as plt
from configs import Config
import matplotlib.image as img
import torch.nn as nn
from torch.nn import init
from tqdm import tqdm
from utils import *
from checkpoint import Checkpoint
from net import ZSSRNet, DownBlock
from torchvision.models.vgg import vgg16


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
        # Acquire meta parameters configuration from configuration class as a class variable
        self.conf = conf

        # Read input image (can be either a numpy array or a path to an image file)
        self.input = img.imread(input_img)
        # print('fuck_lr', self.input.shape)
        # (256, 384, 3)

        # Preprocess the kernels. (see function to see what in includes).
        self.kernels = preprocess_kernels(kernels, conf)

        # The first hr father source is the input (source goes through augmentation to become a father)
        # Later on, if we use gradual sr increments, results for intermediate scales will be added as sources.
        self.hr_fathers_sources = [self.input]
        # self.input.shape: 240, 250, 3

        # We keep the input file name to save the output with a similar name. If array was given rather than path
        # then we use default provided by the configs
        self.file_name = input_img if type(input_img) is str else conf.name

        ##################################DRN##############################
        self.checkpoint = Checkpoint(conf)
        self.device = torch.device('cpu' if conf.cpu else 'cuda')
        self.model = ZSSRNet(input_channels=3)
        # self.optimizer = make_optimizer(conf, self.drnmodel)
        # self.scheduler = make_scheduler(conf, self.optimizer)
        self.dual_models = DownBlock(conf, 2).to(self.device)
        # if not conf.cpu and conf.n_GPUs > 1:
        #     self.model = nn.DataParallel(self.drnmodel, range(conf.n_GPUs))
        #     self.dual_models = self.dataparallel(self.dual_models, range(conf.n_GPUs))
        # self.dual_optimizers = make_dual_optimizer(conf, self.dual_models)
        # self.dual_scheduler = make_dual_scheduler(conf, self.dual_optimizers)
        # self.load(self.checkpoint, pre_train=self.conf.pre_train, cpu=self.conf.cpu)

    # 入口,整个run函数是在不同尺度下和kernel下的大循环
    def run(self):
        # Run gradually on all scale factors (if only one jump then this loop only happens once)
        for self.sf_ind, (sf, self.kernel) in enumerate(zip(self.conf.scale_factors, self.kernels)):
            # verbose
            print('** Start training for sf=', sf, ' **')
            # Relative_sf (used when base change is enabled. this is when input is the output of some previous scale)
            if np.isscalar(sf):
                sf = [sf, sf]
            # sf 默认为[[2.0, 2.0]]
            self.sf = np.array(sf) / np.array(self.base_sf)
            # self.sf [2. 2.]
            self.output_shape = np.uint(np.ceil(np.array(self.input.shape[0:2]) * sf))

            # Initialize network
            self.loss = [None] * self.conf.max_iters
            self.mse, self.mse_rec, self.interp_mse, self.interp_rec_mse, self.mse_steps = [], [], [], [], []
            self.iter = 0
            self.learning_rate = self.conf.learning_rate
            self.learning_rate_change_iter_nums = [0]

            # Train the network
            self.train()  # 训练好几轮，每次都增强,结果保存在 self.train_output

            # Use augmented outputs and back projection to enhance result. Also save the result.
            post_processed_output = self.final_test(self.input)  # 用的是最开始的输入,结果放入hr_fathers_sources
            for i in range(int(math.log(self.conf.scale, 2))):
                post_processed_output = self.final_test(post_processed_output)
            # post_processed_output = self.test(self.input)
            # print('fuck_out', post_processed_output.shape)
            # Keep the results for the next scale factors SR to use as dataset
            self.hr_fathers_sources.append(post_processed_output)

            # In some cases, the current output becomes the new input. If indicated and if this is the right scale to
            # become the new base input. all of these conditions are checked inside the function.
            self.base_change()

            # Save the final output if indicated
            if self.conf.save_results:
                plt.imsave('%s/%sHR.png' %
                           (self.conf.result_path, os.path.basename(self.file_name)[:-6]),
                           post_processed_output, vmin=0, vmax=1)

            # verbose
            print('** Done training for sf=', sf, ' **')

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

    def dataparallel(self, model, gpu_list):
        ngpus = len(gpu_list)
        assert ngpus != 0, "only support gpu mode"
        assert torch.cuda.device_count() >= ngpus, "Invalid Number of GPUs"
        assert isinstance(model, list), "Invalid Type of Dual model"
        for i in range(len(model)):
            if ngpus >= 2:
                model[i] = nn.DataParallel(model[i], gpu_list).cuda()
            else:
                model[i] = model[i].cuda()
        return model

    def father_to_son(self, hr_father):
        lr_son = imresize(hr_father, output_shape=[hr_father.shape[0] / 2, hr_father.shape[1] / 2], kernel=self.kernel)
        return np.clip(lr_son + np.random.randn(*lr_son.shape) * self.conf.noise_std, 0, 1)

    # 在每次迭代前都进行数据增强
    def train(self):
        self.model.cuda()
        self.model.train()
        self.dual_models.cuda()
        self.dual_models.train()
        # vgg = vgg16(pretrained=True)
        # loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        # loss_network.cuda()
        # for param in loss_network.parameters():
        #     param.requires_grad = False
        # self.loss_network = loss_network
        loss = nn.L1Loss()
        # loss = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # sampler 数据增强得到HR-LR pair
        self.hr_fathers_sources = [self.input]  # input 格式
        progress = tqdm(range(self.conf.max_iters))
        for self.iter in progress:  # conf.max_iters
            # Use augmentation from original input image to create current father.
            # If other scale factors were applied before, their result is also used (hr_fathers_in)
            self.model.zero_grad()
            self.dual_models.zero_grad()
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
            # Get lr-son from hr-father 对于输入图像，要得到不同尺度的pair,4x需要经过降采样得到1x和2x,对于2x只需要一次
            # 网络的输入为 h w c
            # print('fuck self.hr_father', self.hr_father.shape)
            # drn网络输入为self.lr_son
            lrs = []
            lr_son = self.hr_father
            lrs.append(lr_son)
            x2 = []
            x4 = []
            sr2lr = []
            # 低分图像真值 和插值输入
            for i in range(int(math.log(self.conf.scale, 2))):
                lr_son = self.father_to_son(lr_son)
                lrs.insert(0, lr_son)  # 小尺度在前面

            # 稠密结果输出有1/2 (n^2+n)个结果:0.5 * (pow(len(self.conf.scale), 2) + len(self.conf.scale))
            for i in range(len(lrs)):
                interpolated = imresize(lrs[i], self.sf, None, self.conf.upscale_method)
                lrx2 = self.model(torch.tensor(interpolated, dtype=torch.float32).cuda())
                x2.append(lrx2)
                interpolated = imresize(lrx2.cpu().detach().numpy(), self.sf, None, self.conf.upscale_method)
                lrx4 = self.model(torch.tensor(interpolated, dtype=torch.float32).cuda())
                x4.append(lrx4)
            # 转tensor
            for i in range(len(lrs)):
                lrs[i] = torch.tensor(lrs[i], dtype=torch.float32).cuda()
            # # D 中间结果
            # for i in range(len(output)):  # 遍历输出结果
            #     if self.conf.scale == 2:
            #         sr2lr_i = self.dual_models(output[i])
            #         sr2lr.append(sr2lr_i.squeeze(0).permute(1, 2, 0))
            #     else:
            #         if self.conf.scale == 4:
            #             sr2lr_i = self.dual_models(output[i])
            #             sr2lr_i = sr2lr_i.squeeze(0).permute(1, 2, 0)
            #             sr2lr_i = self.dual_models(sr2lr_i)
            #             sr2lr.append(sr2lr_i.squeeze(0).permute(1, 2, 0))
            #         else:
            #             if self.conf.scale == 8:
            #                 sr2lr_i = self.dual_models(output[i])
            #                 sr2lr_i = sr2lr_i.squeeze(0).permute(1, 2, 0)
            #                 sr2lr_i = self.dual_models(sr2lr_i)
            #                 sr2lr_i = sr2lr_i.squeeze(0).permute(1, 2, 0)
            #                 sr2lr_i = self.dual_models(sr2lr_i)
            #                 sr2lr.append(sr2lr_i.squeeze(0).permute(1, 2, 0))

            # primary loss:
            loss_primary = torch.zeros(1).cuda()
            for i in range(len(lrs) - 1):
                loss_primary += loss(x2[i], lrs[i + 1])
            for i in range(len(lrs) - 2):
                loss_primary += loss(x4[i], lrs[i + 2])
            # print('fuck_P', output[i].shape, lrs[j].shape)

            # dual loss
            target = self.dual_models(lrs[-1]).squeeze(0).permute(1, 2, 0)
            for i in range(int(math.log(self.conf.scale, 2)) - 1):
                target = self.dual_models(target).squeeze(0).permute(1, 2, 0)
            loss_dual = loss(target, lrs[0])

            # scale loss 除了lr[0]有监督,其他都是无监督
            # loss_cycle = torch.zeros(1).cuda()
            # for i in range(1, len(output) - 1):  # 遍历输出结果
            #     loss_cycle += loss(sr2lr[i], lrs[i])

            # perceptual loss (not good)
            # perc_loss = torch.zeros(1).cuda()
            # interpolated = imresize(lrs[0], [float(self.conf.scale[-1]), float(self.conf.scale[-1])],
            #                         None, self.conf.upscale_method)
            # input = input.permute(2, 0, 1).unsqueeze(0)
            # sr_ = self.model(torch.tensor(interpolated, dtype=torch.float32).cuda()).permute(2, 0, 1).unsqueeze(0)
            # perc_loss += mse_loss(self.loss_network(sr_), self.loss_network(input))

            # compute total loss
            self.loss[
                self.iter] = loss_primary + self.conf.dual_weight * loss_dual  # + self.conf.cycle_weight * loss_cycle  # + 0.006 * perc_loss
            cpu_loss = self.loss[self.iter].data.cpu().numpy()
            progress.set_description(
                "Iter: {iter} Loss: {loss}, Lr: {lr}".format(iter=self.iter, loss=cpu_loss.round(4),
                                                             lr=self.learning_rate))
            if self.iter > 0 and self.iter % 1000 == 0:
                self.learning_rate = self.learning_rate / 10
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.learning_rate

            self.loss[self.iter].backward()
            optimizer.step()

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

    def getTVLoss(self, x):
        x = x.permute(2, 0, 1).unsqueeze(0)
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return (h_tv / count_h + w_tv / count_w) / batch_size

    def tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

    def final_test(self, input):  # input作为测试输入,八种几何变换
        # Run over 8 augmentations of input - 4 rotations and mirror (geometric self ensemble)
        outputs = []
        # The weird range means we only do it once if output_flip is disabled
        # We need to check if scale factor is symmetric to all dimensions, if not we will do 180 jumps rather than 90
        for k in range(0, 1 + 7 * self.conf.output_flip, 1 + int(self.sf[0] != self.sf[1])):
            # Rotate 90*k degrees and mirror flip when k>=4
            test_input = np.rot90(input, k) if k < 4 else np.fliplr(np.rot90(input, k))
            # Apply network on the rotated input,此时不计算损失回传(相当于测试)
            tmp_output = self.test(test_input)
            # Undo the rotation for the processed output (mind the opposite order of the flip and the rotation)
            tmp_output = np.rot90(tmp_output, -k) if k < 4 else np.rot90(np.fliplr(tmp_output), -k)
            # fix SR output with back projection technique for each augmentation
            for bp_iter in range(self.conf.back_projection_iters[self.sf_ind]):
                tmp_output = self.back_projection(tmp_output, input, down_kernel=self.kernel,
                                                  up_kernel=self.conf.upscale_method)
            # save outputs from all augmentations
            outputs.append(tmp_output)

        # Take the median over all 8 outputs
        almost_final_sr = np.median(outputs, 0)

        # Again back projection for the final fused result
        for bp_iter in range(self.conf.back_projection_iters[self.sf_ind]):
            almost_final_sr = self.back_projection(almost_final_sr, input, down_kernel=self.kernel,
                                                   up_kernel=self.conf.upscale_method)

        # Now we can keep the final result (in grayscale case, colors still need to be added, but we don't care
        # because it is done before saving and for every other purpose we use this result)
        self.final_sr = almost_final_sr

        # Add colors to result image in case net was activated only on grayscale
        return self.final_sr

    def base_change(self):
        # If there is no base scale large than the current one get out of here
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

    # 迭代的上下采样,将y_sr降采样与y_lr的差上采样再合并到y_sr上   下->减->上->加
    def back_projection(self, y_sr, y_lr, down_kernel, up_kernel):
        y_sr += imresize(y_lr - imresize(y_sr, output_shape=y_lr.shape[0:2],
                                         kernel=down_kernel), output_shape=y_sr.shape[0:2],
                         kernel=up_kernel)
        return np.clip(y_sr, 0, 1)
