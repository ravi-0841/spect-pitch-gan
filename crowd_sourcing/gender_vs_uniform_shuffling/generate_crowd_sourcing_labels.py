#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:25:48 2020

@author: ravi
"""

import csv
import numpy as np
import shutil
import os

from glob import glob

base_folder = '/home/ravi/Desktop/spect-pitch-gan/crowd_sourcing/gender_vs_uniform_shuffling/neu-ang/'
speech_folder = 'uniformly_shuffled'

fs = sorted(glob(os.path.join(base_folder, speech_folder, '*_neutral.wav')))

existing_numbers = list()
with open(base_folder+'mturk_testing.csv', 'a') as file:
    writer = csv.writer(file)
#    writer.writerow(["media-link", "file_name_src", "file_name_conv"])
    
    for f in fs:
        rand_num = np.random.randint(1e8, 1e9)
        while (rand_num in existing_numbers) or (rand_num+1 in existing_numbers):
            rand_num = np.random.randint(1e8, 1e9)
        existing_numbers += [rand_num, rand_num+1]
        shutil.copy(f, base_folder+'mturk_data/'+str(rand_num)+'.wav')
        base_name = os.path.basename(f)
        file_num = base_name.split('_')[0]
        folder = '/'.join(f.split('/')[:-1]) + '/'
        shutil.copy(folder+file_num+'.wav', base_folder+'mturk_data/' \
                    + str(rand_num+1)+'.wav')
        print(os.path.basename(f), file_num+'.wav')
        writer.writerow([str(rand_num)+'.wav'+','+str(rand_num+1)+'.wav', \
                          f, folder+file_num+'.wav'])
