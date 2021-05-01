# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 17:43:13 2021

@author: vivek.bhadouria
"""
import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as ssim
from PIL import ImageFilter, Image
import glob
import multiprocessing

font      = cv2.FONT_HERSHEY_SIMPLEX
top_left  = (50,80)
font_scale = 1
font_color = (0, 0, 0)
lineType  = 2

base_folder = r'C:\Users\vivek.bhadouria\Documents\Vivek\prj_line_noise'

def per_plane_noise_reduction(color_kernel):
    color_flat = color_kernel.flatten()
    color_flat_nonzero = color_flat[color_flat!=0]
    if len(color_flat_nonzero):
        reconstructed_val = int(np.median(color_flat_nonzero))
    else:
        reconstructed_val = 0 # This approach will fail for high noise density.
    return reconstructed_val

def remove_column_noise(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_copy = img.copy()
    img_copy_temp = img.copy()
    aggregated_columns = np.zeros_like(img)
    pix_to_pad = 1
    img = cv2.copyMakeBorder( img, pix_to_pad, pix_to_pad, pix_to_pad, pix_to_pad, cv2.BORDER_REFLECT)
    column_sums = (img.sum(axis=0).sum(axis=1))
    noisy_column_loc = np.where(column_sums == 0)[0]
    
    # Denoising noisy columns
    for i in noisy_column_loc:
        pixels_buffer = img[:,i-1:i+2,:]
        r_plane = pixels_buffer[:,:,2]
        g_plane = pixels_buffer[:,:,1]
        b_plane = pixels_buffer[:,:,0]
        reconstructed_arr_r = []
        reconstructed_arr_g = []
        reconstructed_arr_b = []
        for j in range(pixels_buffer.shape[0]):
            if j == 0 or j == pixels_buffer.shape[0] - 1: # Do not process border pixels
                continue
            # Starting extracting the kernel
            reconstructed_val_r = per_plane_noise_reduction(r_plane[j-1:j+2, :])
            reconstructed_arr_r.append(reconstructed_val_r)

            reconstructed_val_g = per_plane_noise_reduction(g_plane[j-1:j+2, :])
            reconstructed_arr_g.append(reconstructed_val_g)

            reconstructed_val_b = per_plane_noise_reduction(b_plane[j-1:j+2, :])
            reconstructed_arr_b.append(reconstructed_val_b)
        reconstructed_arr_r = np.asarray(reconstructed_arr_r, dtype=np.uint8).transpose()
        # reconstructed_arr_r = np.expand_dims(reconstructed_arr_r, axis=1)
        aggregated_columns[:,i-1,2] = reconstructed_arr_r
        aggregated_columns[:,i-1,1] = reconstructed_arr_g
        aggregated_columns[:,i-1,0] = reconstructed_arr_b
    out = cv2.add(img_copy,aggregated_columns)
    # out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    out_pil = Image.fromarray(out)
    sharpened_img = out_pil.filter(ImageFilter.SHARPEN)
    sharpened_img = np.asarray(sharpened_img, dtype=np.uint8)
    # ssim_const = ssim(img_copy, out, multichannel=True, data_range=out.max() - out.min())
    # print(ssim_const)

    # Change only those columns which were noise corrupted
    for i in noisy_column_loc:
        img_copy_temp[:,i-1,:] = sharpened_img[:,i-1,:]
        
    # Read original image for generating metrics
    orig_file_path = filepath.replace('noisy_frames', 'extracted_frames')
    orig_img = cv2.imread(orig_file_path, cv2.IMREAD_COLOR)
    
    ssim_noisy = ssim(orig_img, img_copy, multichannel=True)
    ssim_corrected_img = ssim (orig_img, img_copy_temp, multichannel=True)
    
    psnr_noisy = cv2.PSNR(orig_img, img_copy)
    psnr_corrected_img = cv2.PSNR(orig_img, img_copy_temp)

    text_to_print_noisy_lines = 'Noisy Lines: '+str(len(noisy_column_loc)).zfill(2)

    text_to_print_ssim = 'Noisy SSIM: '+('{:0.4f}'.format(ssim_noisy))+' '+'Reconstructed SSIM: '+('{:0.4f}'.format(ssim_corrected_img))
    text_to_print_psnr = 'Noisy PSNR: '+('{:0.2f}'.format(psnr_noisy))+' '+'Reconstructed PSNR: '+('{:0.2f}'.format(psnr_corrected_img))

    # text_to_print_ssim = 'Noisy SSIM: '+str(round(ssim_noisy, 4)).zfill(4)+' '+'Reconstructed SSIM: '+str(round(ssim_corrected_img, 4))
    # text_to_print_psnr = 'Noisy PSNR: '+str(round(psnr_noisy, 2))+' '+'Reconstructed PSNR: '+str(round(psnr_corrected_img, 2))
    
    cv2.putText(img_copy_temp, text_to_print_noisy_lines, (50, 80), font, font_scale, font_color, lineType)
    cv2.putText(img_copy_temp, text_to_print_ssim, (50, 110), font, font_scale, font_color, lineType)
    cv2.putText(img_copy_temp, text_to_print_psnr, (50, 140), font, font_scale, font_color, lineType)
    
    modified_img_path = filepath.replace('noisy_frames','noise_reduced_frames')
    cv2.imwrite(modified_img_path, img_copy_temp)

if __name__ == "__main__":
    p = multiprocessing.Pool(multiprocessing.cpu_count())

    img_list = glob.glob(os.path.join(base_folder, 'noisy_frames', '*.png'))
    p.map(remove_column_noise, img_list)
    
