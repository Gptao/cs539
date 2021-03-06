# import math
# import cv2
# from math import pi, sin, cos
# from cv2 import warpPerspective, INTER_CUBIC
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.image as img  # 图像的颜色像素值在0和1之间
# import skimage.io as io
# '''cv2.imread()读取彩色图像后得到的格式是BGR格式，像素值范围在0~255之间，通道格式为(H,W,C)，
#     想要显示RGB类型的图像要进行一步格式转换,其他都是RGB格式'''
# def PSNR(pred, gt, shave_border=0):
#     height, width = pred.shape[:2]
#     pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
#     gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
#     imdff = pred - gt
#     rmse = math.sqrt(np.mean(imdff ** 2))
#     if rmse == 0:
#         return 100
#     return 20 * math.log10(255.0 / rmse)
#
# a = cv2.imread('Set5/image_SRF_4/img_005_SRF_4_LR.png')
# # cv2.imshow('test', a)
# # b=cv2.resize(a,(int(a.shape[1]/2),int(a.shape[0]/2)),interpolation=cv2.INTER_CUBIC)
# # # size ： shape[1]*shape[2]*shape[3]
# # cv2.imshow('resize', b)
# print(a)
# print("#######################################")
# b = Image.open('Set5/image_SRF_4/img_005_SRF_4_LR.png')
# print(np.array(b))
# print("#######################################")
# c = img.imread('Set5/image_SRF_4/img_005_SRF_4_LR.png')
# print(np.array(c))
# print("#######################################")
# d=io.imread('Set5/image_SRF_4/img_005_SRF_4_LR.png')
# print(d)
# print('#######################################')
##########################################################################
# 灰度图和rgb转换
# from PIL import Image
# import numpy as np
#
# L_path = 'Set14/image_SRF_4/img_003_SRF_4_LR.png'
# L_image = Image.open(L_path)
# origin = np.array(L_image)
# print(origin.shape)
# out = L_image.convert("RGB")
# out.save('Set14/image_SRF_2/img_003_SRF_2_LR.png')

# img=Image.open('HR/image_SRF_2/Set14/img_003_SRF_2_HR.png')
# a=np.array(img)
# print(a.shape)
'''
calculate the PSNR and SSIM.
same as MATLAB's results
'''

# import os
# import math
# import numpy as np
# import cv2
# import glob
#
#
# def main():
#     # Configurations
#
#     # GT - Ground-truth;
#     # Gen: Generated / Restored / Recovered images
#     folder_GT = 'HR/image_SRF_4/Set5'
#     folder_Gen = 'results/set5_x4_origin'
#
#     crop_border = 4
#     suffix = ''  # suffix for Gen images
#     test_Y = True  # True: test Y channel only; False: test RGB channels
#
#     PSNR_all = []
#     SSIM_all = []
#     img_list = sorted(glob.glob(folder_GT + '/*'))
#
#     if test_Y:
#         print('Testing Y channel.')
#     else:
#         print('Testing RGB channels.')
#
#     for i, img_path in enumerate(img_list):
#         base_name = os.path.splitext(os.path.basename(img_path))[0]
#         im_GT = cv2.imread(img_path) / 255.
#         im_Gen = cv2.imread(os.path.join(folder_Gen, base_name + suffix + '.png')) / 255.
#
#         if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
#             im_GT_in = bgr2ycbcr(im_GT)
#             im_Gen_in = bgr2ycbcr(im_Gen)
#         else:
#             im_GT_in = im_GT
#             im_Gen_in = im_Gen
#
#         # crop borders
#         if im_GT_in.ndim == 3:
#             cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
#             cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
#         elif im_GT_in.ndim == 2:
#             cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
#             cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
#         else:
#             raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))
#
#         # calculate PSNR and SSIM
#         PSNR = calculate_psnr(cropped_GT * 255, cropped_Gen * 255)
#
#         SSIM = calculate_ssim(cropped_GT * 255, cropped_Gen * 255)
#         print('{:3d} - {:25}. \tPSNR: {:.6f} dB, \tSSIM: {:.6f}'.format(
#             i + 1, base_name, PSNR, SSIM))
#         PSNR_all.append(PSNR)
#         SSIM_all.append(SSIM)
#     print('Average: PSNR: {:.6f} dB, SSIM: {:.6f}'.format(
#         sum(PSNR_all) / len(PSNR_all),
#         sum(SSIM_all) / len(SSIM_all)))
#
#
# def calculate_psnr(img1, img2):
#     # img1 and img2 have range [0, 255]
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     mse = np.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return float('inf')
#     return 20 * math.log10(255.0 / math.sqrt(mse))
#
#
# def ssim(img1, img2):
#     C1 = (0.01 * 255) ** 2
#     C2 = (0.03 * 255) ** 2
#
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     kernel = cv2.getGaussianKernel(11, 1.5)
#     window = np.outer(kernel, kernel.transpose())
#
#     mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
#     mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
#     mu1_sq = mu1 ** 2
#     mu2_sq = mu2 ** 2
#     mu1_mu2 = mu1 * mu2
#     sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
#     sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
#     sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
#
#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
#                                                             (sigma1_sq + sigma2_sq + C2))
#     return ssim_map.mean()
#
#
# def calculate_ssim(img1, img2):
#     '''calculate SSIM
#     the same outputs as MATLAB's
#     img1, img2: [0, 255]
#     '''
#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')
#     if img1.ndim == 2:
#         return ssim(img1, img2)
#     elif img1.ndim == 3:
#         if img1.shape[2] == 3:
#             ssims = []
#             for i in range(3):
#                 ssims.append(ssim(img1, img2))
#             return np.array(ssims).mean()
#         elif img1.shape[2] == 1:
#             return ssim(np.squeeze(img1), np.squeeze(img2))
#     else:
#         raise ValueError('Wrong input image dimensions.')
#
#
# def bgr2ycbcr(img, only_y=True):
#     '''same as matlab rgb2ycbcr
#     only_y: only return Y channel
#     Input:
#         uint8, [0, 255]
#         float, [0, 1]
#     '''
#     in_img_type = img.dtype
#     img.astype(np.float32)
#     if in_img_type != np.uint8:
#         img *= 255.
#     # convert
#     if only_y:
#         rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
#     else:
#         rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
#                               [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
#     if in_img_type == np.uint8:
#         rlt = rlt.round()
#     else:
#         rlt /= 255.
#     return rlt.astype(in_img_type)
#
#
# if __name__ == '__main__':
#     main()
# -*- coding: utf-8 -*-
# import logging
# import time
# from multiprocessing.dummy import Pool as ThreadPool
#
# from tqdm import tqdm
#
#
# def get_logger(name):
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.DEBUG)
#     stream_handler = logging.StreamHandler()
#     stream_handler.setLevel(logging.DEBUG)
#     formatter = logging.Formatter(
#         '%(asctime)s - %(name)s [%(levelname)s] %(message)s')
#     stream_handler.setFormatter(formatter)
#     logger.addHandler(stream_handler)
#     return logger
#
#
# def process(item):
#     log = get_logger(str(item[0]))
#     log.info("sum: {sum} ".format(sum=item[0]+item[1]))
#     time.sleep(5)
#
#
# items = [[1, 2], [3, 4], [5, 6], [7, 8]]
# pool = ThreadPool()
# pool.map(process, items)
# pool.close()
# pool.join()
# import threading
# import tkinter as tk
# from time import ctime, sleep
#
# # 创建主窗口
# window = tk.Tk()
# window.title('测试')
# window.geometry('630x200')
#
#
# def music():
#     for i in range(2):
#         print("I was listening to music %s" % ctime())
#         sleep(1)
#
#
# def move():
#     for i in range(2):
#         print("I was at the movie  %s" % ctime())
#         sleep(1)
#
#
# def test():
#     # 多线程
#     threads = []
#     t1 = threading.Thread(target=music)
#     threads.append(t1)
#     t2 = threading.Thread(target=move)
#     threads.append(t2)
#     for t in threads:
#         t.setDaemon(True)
#         t.start()
#
#
# btn_download = tk.Button(window, text='启动', command=test)
# btn_download.place(x=400, y=150)
#
# window.mainloop()
import matplotlib.image as img
from imresize import imresize
import numpy as np


def father_to_son(hr_father):
    sf = np.array([1.0, 1.5])
    lr_son = imresize(hr_father, 1.0 / sf, kernel='cubic')
    print(lr_son.shape)
    lr_son = imresize(lr_son, sf, kernel='cubic')
    print(lr_son.shape)
    return np.clip(lr_son, 0, 1)


if __name__ == '__main__':
    testimg = img.imread('Set14/image_SRF_4/img_003_SRF_4_LR.png')
    son = father_to_son(testimg)
    print(testimg.shape)
