# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 17:17:38 2021

@author: vivek.bhadouria
"""

# Non local means

import glob
import numpy as np
import random
import cv2
import os
import multiprocessing

base_folder = r'C:\Users\vivek.bhadouria\Documents\Vivek\prj_line_noise'

def generate_random_list(start_sequence, end_sequence, num_element):
    return [random.randrange(start_sequence, end_sequence, 1) for i in range(num_element)]

def add_column_noise(input_image):
    img = cv2.imread(input_image, cv2.IMREAD_COLOR)
    img_row, img_col, _ = img.shape
    # Generate noisy column
    noisy_col = np.zeros([img_row, 1, 3])
    # Generate column list where noise has to be added. Assume 1% of columns are corrupted i.e. [19,21]
    num_random_columns = random.randint(19,21)
    res = generate_random_list(3, 1917, num_random_columns)
    # Fuse all the noisy columns into image
    for i in res:
        img[:, i-1:i, :] = noisy_col
    modified_img_path = input_image.replace('extracted_frames','noisy_frames')        
    cv2.imwrite(modified_img_path, img)

if __name__ == "__main__":
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    
    img_list = glob.glob(os.path.join(base_folder, 'extracted_frames', '*.png'))
    p.map(add_column_noise, img_list)
