# coding=utf-8
from math import pi, sin, cos
from cv2 import warpPerspective, INTER_CUBIC
from imresize import imresize
from shutil import copy
from time import strftime, localtime
import os
import glob
from scipy.ndimage import measurements, interpolation
from scipy.io import loadmat
import math
import time
import random
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs


def random_augment(ims,
                   base_scales=None,
                   leave_as_is_probability=0.2,
                   no_interpolate_probability=0.3,
                   min_scale=0.5,
                   max_scale=1.0,
                   allow_rotation=True,
                   scale_diff_sigma=0.01,
                   shear_sigma=0.01,
                   crop_size=128):
    # Determine which kind of augmentation takes place according to probabilities
    random_chooser = np.random.rand()

    # Option 1: No augmentation, return the original image
    if random_chooser < leave_as_is_probability:
        mode = 'leave_as_is'

    # Option 2: Only non-interpolated augmentation, which means 8 possible augs (4 rotations X 2 mirror flips)
    elif leave_as_is_probability < random_chooser < leave_as_is_probability + no_interpolate_probability:
        mode = 'no_interp'

    # Option 3: Affine transformation (uses interpolation)
    else:
        mode = 'affine'

    # If scales not given, calculate them according to sizes of images. This would be suboptimal, because when scales
    # are not integers, different scales can have the same image shape.
    if base_scales is None:
        base_scales = [np.sqrt(np.prod(im.shape) / np.prod(ims[0].shape)) for im in ims]

    # In case scale is a list of scales with take the smallest one to be the allowed minimum
    max_scale = np.min([max_scale])

    # Determine a random scale by probability
    if mode == 'leave_as_is':
        scale = 1.0
    else:
        scale = np.random.rand() * (max_scale - min_scale) + min_scale

    # The image we will use is the smallest one that is bigger than the wanted scale
    # (Using a small value overlap instead of >= to prevent float issues)
    scale_ind, base_scale = next((ind, np.min([base_scale])) for ind, base_scale in enumerate(base_scales)
                                 if np.min([base_scale]) > scale - 1.0e-6)
    im = ims[scale_ind]

    # Next are matrices whose multiplication will be the transformation. All are 3x3 matrices.

    # First matrix shifts image to center so that crop is in the center of the image
    shift_to_center_mat = np.array([[1, 0, - im.shape[1] / 2.0],
                                    [0, 1, - im.shape[0] / 2.0],
                                    [0, 0, 1]])

    shift_back_from_center = np.array([[1, 0, im.shape[1] / 2.0],
                                       [0, 1, im.shape[0] / 2.0],
                                       [0, 0, 1]])
    # Keeping the transform interpolation free means only shifting by integers
    if mode != 'affine':
        shift_to_center_mat = np.round(shift_to_center_mat)
        shift_back_from_center = np.round(shift_back_from_center)

    # Scale matrix. if affine, first determine global scale by probability, then determine difference between x scale
    # and y scale by gaussian probability.
    if mode == 'affine':
        scale /= base_scale
        scale_diff = np.random.randn() * scale_diff_sigma
    else:
        scale = 1.0
        scale_diff = 0.0
    # In this matrix we also incorporate the possibility of mirror reflection (unless leave_as_is).
    if mode == 'leave_as_is' or not allow_rotation:
        reflect = 1
    else:
        reflect = np.sign(np.random.randn())

    scale_mat = np.array([[reflect * (scale + scale_diff / 2), 0, 0],
                          [0, scale - scale_diff / 2, 0],
                          [0, 0, 1]])

    # Shift matrix, this actually creates the random crop
    shift_x = np.random.rand() * np.clip(scale * im.shape[1] - crop_size, 0, 9999)
    shift_y = np.random.rand() * np.clip(scale * im.shape[0] - crop_size, 0, 9999)
    shift_mat = np.array([[1, 0, - shift_x],
                          [0, 1, - shift_y],
                          [0, 0, 1]])

    # Keeping the transform interpolation free means only shifting by integers
    if mode != 'affine':
        shift_mat = np.round(shift_mat)

    # Rotation matrix angle. if affine, set a random angle. if no_interp then theta can only be pi/2 times int.
    if mode == 'affine':
        theta = np.random.rand() * 2 * pi
    elif mode == 'no_interp':
        theta = np.random.randint(4) * pi / 2
    else:
        theta = 0
    if not allow_rotation:
        theta = 0

    # Rotation matrix structure
    rotation_mat = np.array([[cos(theta), sin(theta), 0],
                             [-sin(theta), cos(theta), 0],
                             [0, 0, 1]])

    # Shear Matrix, only for affine transformation.
    if mode == 'affine':
        shear_x = np.random.randn() * shear_sigma
        shear_y = np.random.randn() * shear_sigma
    else:
        shear_x = shear_y = 0
    shear_mat = np.array([[1, shear_x, 0],
                          [shear_y, 1, 0],
                          [0, 0, 1]])

    # Create the final transformation by multiplying all the transformations.
    transform_mat = (shift_back_from_center
                     .dot(shift_mat)
                     .dot(shear_mat)
                     .dot(rotation_mat)
                     .dot(scale_mat)
                     .dot(shift_to_center_mat))

    # Apply transformation to image and return the transformed image clipped between 0-1
    return np.clip(warpPerspective(im, transform_mat, (crop_size, crop_size), flags=INTER_CUBIC), 0, 1)


# 迭代的上下采样,将y_sr降采样与y_lr的差上采样再合并到y_sr上   下->减->上->加
# imresize(hr_father, output_shape=[hr_father.shape[0] / 2, hr_father.shape[1] / 2], kernel=self.kernel)
def back_projection(y_sr, y_lr, down_kernel, up_kernel):
    y_sr += imresize(y_lr - imresize(y_sr, output_shape=y_lr.shape[0:2],
                                     kernel=down_kernel), output_shape=y_sr.shape[0:2],
                     kernel=up_kernel)
    return np.clip(y_sr, 0, 1)


def preprocess_kernels(kernels, conf):
    # Load kernels if given files. if not just use the downscaling method from the configs.
    # output is a list of kernel-arrays or a a list of strings indicating downscaling method.
    # In case of arrays, we shift the kernels (see next function for explanation why).
    # Kernel is a .mat file (MATLAB) containing a variable called 'Kernel' which is a 2-dim matrix.
    if kernels is not None:
        return [kernel_shift(loadmat(kernel)['Kernel'], sf)
                for kernel, sf in zip(kernels, conf.scale_factors)]
    else:
        return [conf.downscale_method] * len(conf.scale_factors)


def kernel_shift(kernel, sf):
    # There are two reasons for shifting the kernel:
    # 1. Center of mass is not in the center of the kernel which creates ambiguity. There is no possible way to know
    #    the degradation process included shifting so we always assume center of mass is center of the kernel.
    # 2. We further shift kernel center so that top left result pixel corresponds to the middle of the sfXsf first
    #    pixels. Default is for odd size to be in the middle of the first pixel and for even sized kernel to be at the
    #    top left corner of the first pixel. that is why different shift size needed between odd and even size.
    # Given that these two conditions are fulfilled, we are happy and aligned, the way to test it is as follows:
    # The input image, when interpolated (regular bicubic) is exactly aligned with ground truth.

    # First calculate the current center of mass for the kernel
    current_center_of_mass = measurements.center_of_mass(kernel)

    # The second term ("+ 0.5 * ....") is for applying condition 2 from the comments above
    wanted_center_of_mass = np.array(kernel.shape) / 2 + 0.5 * (np.array(sf) - (np.array(kernel.shape) % 2))

    # Define the shift vector for the kernel shifting (x,y)
    shift_vec = wanted_center_of_mass - current_center_of_mass

    # Before applying the shift, we first pad the kernel so that nothing is lost due to the shift
    # (biggest shift among dims + 1 for safety)
    kernel = np.pad(kernel, np.int(np.ceil(np.max(np.abs(shift_vec)))) + 1, 'constant')

    # Finally shift the kernel and return
    return interpolation.shift(kernel, shift_vec)


def prepare_result_dir(conf):
    # Create results directory
    if conf.create_results_dir:
        conf.result_path += '/' + conf.name + strftime('_%b_%d_%H_%M_%S', localtime())
        os.makedirs(conf.result_path)

    # Put a copy of all *.py files in results path, to be able to reproduce experimental results
    if conf.create_code_copy:
        local_dir = os.path.dirname(__file__)
        for py_file in glob.glob(local_dir + '/*.py'):
            copy(py_file, conf.result_path)

    return conf.result_path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() == 1:
        torch.cuda.manual_seed(seed)
    else:
        torch.cuda.manual_seed_all(seed)


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def make_optimizer(opt, my_model):
    # filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
    # 该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判断，然后返回 True 或 False，最后将返回 True 的元素放到新列表中。
    trainable = filter(lambda x: x.requires_grad, my_model.parameters())
    optimizer_function = optim.Adam
    kwargs = {
        'betas': (opt.beta1, opt.beta2),
        'eps': opt.epsilon
    }
    kwargs['lr'] = opt.lr
    kwargs['weight_decay'] = opt.weight_decay

    return optimizer_function(trainable, **kwargs)


def make_dual_optimizer(opt, dual_models):
    dual_optimizers = []
    for dual_model in dual_models:
        '''
        params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
        lr (float, 可选) – 学习率（默认：1e-3）
        betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
        eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
        weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）
        '''
        temp_dual_optim = torch.optim.Adam(
            params=dual_model.parameters(),
            lr=opt.lr,
            betas=(opt.beta1, opt.beta2),
            eps=opt.epsilon,
            weight_decay=opt.weight_decay)
        dual_optimizers.append(temp_dual_optim)

    return dual_optimizers


def make_scheduler(opt, my_optimizer):
    scheduler = lrs.CosineAnnealingLR(
        my_optimizer,
        float(opt.max_iters),
        eta_min=opt.min_learning_rate
    )

    return scheduler


def make_dual_scheduler(opt, dual_optimizers):
    dual_scheduler = []
    for i in range(len(dual_optimizers)):
        scheduler = lrs.CosineAnnealingLR(
            dual_optimizers[i],
            float(opt.max_iters),
            eta_min=opt.min_learning_rate
        )
        dual_scheduler.append(scheduler)

    return dual_scheduler


def init_model(args):
    # Set the templates here
    if args.model.find('DRN-S') >= 0:
        if args.scale == 4:
            args.n_blocks = 30
            args.n_feats = 16
        elif args.scale == 8:
            args.n_blocks = 30
            args.n_feats = 8
        else:
            print('Use defaults n_blocks and n_feats.')
        args.dual = True

    if args.model.find('DRN-L') >= 0:
        if args.scale == 4:
            args.n_blocks = 40
            args.n_feats = 20
        elif args.scale == 8:
            args.n_blocks = 36
            args.n_feats = 10
        else:
            print('Use defaults n_blocks and n_feats.')
        args.dual = True


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
