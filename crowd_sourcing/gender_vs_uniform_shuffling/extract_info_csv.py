#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:38:10 2020

@author: ravi
"""

import pandas as pd
import numpy as np
import pylab

emo_pair = "neu-ang"

file = r'/home/ravi/Desktop/pitch-lddmm-spect/validation_crowd_sourcing/' \
        +emo_pair+ '/'+emo_pair+'_batch_results.csv'

df_resp = pd.read_csv(file)
df_resp = df_resp[df_resp.WorkerId != 'AL3UGBNTPA0I7'] 

unique_links = list(df_resp['Input.media-link'].unique())

emo = {}
for i in unique_links:
    responses = list(df_resp.loc[df_resp['Input.media-link'] == i]['Answer.Emotion'])
    emo[i] = responses.count('angry') / len(responses)
    
mos = {}
for i in unique_links:
    responses = list(df_resp.loc[df_resp['Input.media-link'] == i]['Answer.MOS Score'])
    mos[i] = np.mean(responses)

file = r'/home/ravi/Desktop/pitch-lddmm-spect/validation_crowd_sourcing/' \
        +emo_pair+'/mturk_testing.csv'

df_src = pd.read_csv(file)

no_mod_emo = list()
smoothed_emo = list()
gender_emo = list()
deep_net_emo = list()

no_mod_mos = list()
smoothed_mos = list()
gender_mos = list()
deep_net_mos = list()

for i in unique_links:
    src = df_src.loc[df_src['media-link'] == i]['file_name_src'].to_list()
    src = src[0]
    if 'smoothed' in src:
        smoothed_emo.append(emo[i])
        smoothed_mos.append(mos[i])
    elif 'deeper_net' in src:
        deep_net_emo.append(emo[i])
        deep_net_mos.append(mos[i])
    elif 'gender' in src:
        gender_emo.append(emo[i])
        gender_mos.append(mos[i])
    else:
        no_mod_emo.append(emo[i])
        no_mod_mos.append(mos[i])

print('Smoothed: EMO - {}, MOS- {}'.format(np.mean(smoothed_emo), np.mean(smoothed_mos)))
print('Deeper: EMO - {}, MOS - {}'.format(np.mean(deep_net_emo), np.mean(deep_net_mos)))
print('Gender: EMO - {}, MOS - {}'.format(np.mean(gender_emo), np.mean(gender_mos)))
print('No mod: EMO - {}, MOS - {}'.format(np.mean(no_mod_emo), np.mean(no_mod_mos)))

pylab.figure()
pylab.subplot(121), pylab.boxplot([smoothed_emo, deep_net_emo, gender_emo, no_mod_emo], \
             labels=['Smoothed', 'Deep-Net', 'Gender', 'Original'])
pylab.title("Emotional Saliency (Neutral-Angry)")
pylab.subplot(122), pylab.boxplot([smoothed_mos, deep_net_mos, gender_mos, no_mod_mos], \
             labels=['Smoothed', 'Deep-Net', 'Gender', 'Original'])
pylab.title("MOS (Neutral-Angry)")
