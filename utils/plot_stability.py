#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 16:41:32 2021

@author: ravi
"""
import numpy as np
import pylab
import re
from matplotlib import rc

file = open('/home/ravi/Desktop/CycleGAN_stability_NA.txt', 'r')
gen_loss = []
disc_loss = []
for line in file:
    if "Discriminator" in line:
        n = re.findall(r"[-+]?\d*\.\d+e-\d+|[-+]?\d*\.\d+|\d+", line)
        gen_loss.append(float(n[1]))
        disc_loss.append(float(n[2]))

file.close()
gd_diff_vanilla = np.abs(np.asarray(gen_loss) - np.asarray(disc_loss))


gen_loss = []
disc_loss = []
file_name = '/home/ravi/Desktop/VCGAN_stability_NA.log'

file = open(file_name, 'r')
for line in file:
    if "Generator" in line:
        n = re.findall(r"[-+]?\d*\.\d+e-\d+|[-+]?\d*\.\d+|\d+", line)
        gen_loss.append(float(n[0]))
file.close()

file = open(file_name, 'r')
for line in file:
    if "Discriminator" in line:
        n = re.findall(r"[-+]?\d*\.\d+e-\d+|[-+]?\d*\.\d+|\d+", line)
        disc_loss.append(float(n[0]))
        
gd_diff_variation = np.abs(np.asarray(gen_loss) - np.asarray(disc_loss))

kernel_size = 1
kernel = np.ones((kernel_size,)) / kernel_size

smoothed_variation = np.convolve(gd_diff_variation, kernel, mode='same')
smoothed_vanilla = np.convolve(gd_diff_vanilla, kernel, mode='same')

rc('font', weight='bold')
rc('axes', linewidth=2)
pylab.figure(figsize=(13,9))
pylab.plot(smoothed_variation, label="Vanilla Cycle-GAN", linewidth=3.5)
pylab.plot(smoothed_vanilla, label="Variational Cycle-GAN", linewidth=3.5)
pylab.ylim((-0.2, 1.2)), pylab.xticks(size=20), pylab.yticks(size=20)
pylab.grid(), pylab.legend(loc=1, prop={'size':18, 'weight':'bold'})