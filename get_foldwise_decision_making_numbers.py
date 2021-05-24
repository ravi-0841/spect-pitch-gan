#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 22 15:34:12 2021

@author: ravi
"""

import pylab
import re

emo_pair = 'neu-ang'
fold = 5

f0_loss = []
ec_loss = []

for r in range(1,5):

    f0_loss.append([])
    ec_loss.append([])

    f = open('/home/ravi/Desktop/txt_files/NA_fold_{}_run_{}.txt'.format(fold, r), 'r')
    for line in f:
        if 'epoch' in line:
            try:
                line = next(f)
                line = line.strip('\n')
                f0_mu_std = re.findall(r"[-+]?\d*\.\d+e-\d+|[-+]?\d*\.\d+|\d+", line)
                f0_loss[-1].append(float(f0_mu_std[0]))
                line = next(f)
                line = line.strip('\n')
                ec_mu_std = re.findall(r"[-+]?\d*\.\d+e-\d+|[-+]?\d*\.\d+|\d+", line)
                ec_loss[-1].append(float(ec_mu_std[0]))
            except Exception as ex:
                f0_loss[-1].append(100000)
                ec_loss[-1].append(100)
    
for r in range(4):
        pylab.plot(ec_loss[r][0], f0_loss[r][0], 'r.')
        pylab.plot(ec_loss[r][1], f0_loss[r][1], 'g.')
        pylab.plot(ec_loss[r][2], f0_loss[r][2], 'b.')
        pylab.plot(ec_loss[r][3], f0_loss[r][3], 'k.')
        
        pylab.annotate('run {}'.format(r+1), xy=(ec_loss[r][0], f0_loss[r][0]))
        pylab.annotate('run {}'.format(r+1), xy=(ec_loss[r][1], f0_loss[r][1]))
        pylab.annotate('run {}'.format(r+1), xy=(ec_loss[r][2], f0_loss[r][2]))
        pylab.annotate('run {}'.format(r+1), xy=(ec_loss[r][3], f0_loss[r][3]))
        
