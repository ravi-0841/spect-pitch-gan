#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 14:26:44 2020

@author: ravi
"""

import pandas as pd
import numpy as np
import pylab

emo_pair_list   = ["neu-sad"]

emo_dict        = {'neu-ang':'angry', 'neu-hap':'happy', 'neu-sad':'sad'}

pg_emo_mu       = list()
pg_emo_var      = list()
pg_mos_mu       = list()
pg_mos_var      = list()

cg_emo_mu       = list()
cg_emo_var      = list()
cg_mos_mu       = list()
cg_mos_var      = list()

ref_pg_mos_mu   = list()
ref_pg_mos_var  = list()

ref_cg_mos_mu   = list()
ref_cg_mos_var  = list()

for emo_pair in emo_pair_list:

    print(emo_pair)
    target          = emo_dict[emo_pair]
    file = r'/home/ravi/Desktop/pitch-gan/pitch-lddmm-gan/test_crowd_sourcing/' \
            +'vesus/'+emo_pair+ '/'+emo_pair+'_batch_results_full.csv'
    
    df_resp         = pd.read_csv(file)
    df_resp         = df_resp[df_resp.WorkerId != 'AL3UGBNTPA0I7']
    df_resp         = df_resp[df_resp.WorkerId != 'A3KPQ7L5FS8SD6']
    df_resp         = df_resp[df_resp.WorkerId != 'A2CK0OXMPOR9LE']
    df_resp         = df_resp[df_resp.WorkerId != 'A37WXDYYT7RCZ0']
    df_resp         = df_resp[df_resp.WorkerId != 'A3RHGIMIWFXPJ7']
    
    unique_links = list(df_resp['Input.media-link'].unique())
    
    emo             = {}
    for i in unique_links:
        responses   = list(df_resp.loc[df_resp['Input.media-link'] \
                                       == i]['Answer.Emotion'])
        emo[i]      = responses.count(target) / len(responses)
       
    mos_A           = {}
    mos_B           = {}
    mos             = {}
    for i in unique_links:
        responses_A = list(df_resp.loc[df_resp['Input.media-link'] == i]['Answer.MOS_A'])
        responses_B = list(df_resp.loc[df_resp['Input.media-link'] == i]['Answer.MOS_B'])
        mos_A[i]    = np.mean(np.asarray(responses_A))
        mos_B[i]    = np.mean(np.asarray(responses_B))
        mos[i]      = np.max(np.asarray(responses_A)) - \
                            np.max(np.asarray(responses_B))
    
    file = r'/home/ravi/Desktop/pitch-gan/pitch-lddmm-gan/test_crowd_sourcing/' \
            +'vesus/'+emo_pair+'/mturk_testing_full.csv'
    
    df_src          = pd.read_csv(file)
    
    pair_gan_emo    = list()
    cycle_gan_emo   = list()
    
    pair_gan_mos    = list()
    cycle_gan_mos   = list()
    
    ref_pg_mos      = list()
    ref_cg_mos      = list()
    
    for i in unique_links:
        src         = df_src.loc[df_src['media-link'] \
                                 == i]['file_name_src'].to_list()
        if len(src)>0:
            try:
                src     = src[0]
                if 'pairGAN' in src:
                    pair_gan_emo.append(emo[i])
                    pair_gan_mos.append(mos_B[i])
                    ref_pg_mos.append(mos_A[i])
                elif 'cycle_gan' in src:
                    cycle_gan_emo.append(emo[i])
                    cycle_gan_mos.append(mos_B[i])
                    ref_cg_mos.append(mos_A[i])
    
            except Exception as e:
                print(e, src)
    
    print('Pair-GAN : EMO - {}, MOS- {}'.format(np.mean(pair_gan_emo), np.mean(pair_gan_mos)))
    print('Cycle-GAN : EMO - {}, MOS - {}'.format(np.mean(cycle_gan_emo), np.mean(cycle_gan_mos)))
    
    pg_emo_mu.append(np.mean(pair_gan_emo))
    pg_emo_var.append(np.std(pair_gan_emo))
    pg_mos_mu.append(np.mean(pair_gan_mos))
    pg_mos_var.append(np.std(pair_gan_mos))
    ref_pg_mos_mu.append(np.mean(ref_pg_mos))
    ref_pg_mos_var.append(np.std(ref_pg_mos))
    
    cg_emo_mu.append(np.mean(cycle_gan_emo))
    cg_emo_var.append(np.std(cycle_gan_emo))
    cg_mos_mu.append(np.mean(cycle_gan_mos))
    cg_mos_var.append(np.std(cycle_gan_mos))
    ref_cg_mos_mu.append(np.mean(ref_cg_mos))
    ref_cg_mos_var.append(np.std(ref_cg_mos))

ref_mos_mu = list((np.asarray(ref_pg_mos_mu) \
                   + np.asarray(ref_cg_mos_mu))/2)
ref_mos_var = list((np.asarray(ref_pg_mos_var) \
                    + np.asarray(ref_cg_mos_var))/2)

barWidth=0.3
r1 = np.arange(len(pg_emo_mu))
r2 = [x + barWidth for x in r1]
 
# Emotional Saliency plot
pylab.figure()
pylab.bar(r1, pg_emo_mu, width = barWidth, color = 'blue', \
          edgecolor = 'black', yerr=pg_emo_var, capsize=7, \
          label='Pair-GAN')

pylab.bar(r2, cg_emo_mu, width = barWidth, color = 'cyan', \
        edgecolor = 'black', yerr=cg_emo_var, capsize=7, label='Cycle-GAN')

pylab.xticks([r + 0.5*barWidth for r in range(len(pg_emo_mu))], \
              ['Neu-Ang', 'Neu-Hap', 'Neu-Sad'])
pylab.ylabel('Emotional Saliency')
pylab.legend(loc=2), pylab.title('VESUS Emotional Saliency')


# MOS score plot
barWidth=0.25
r1 = np.arange(len(pg_emo_mu))
r2 = [x + barWidth for x in r1]
r3 = [x + 2*barWidth for x in r1]

pylab.figure()
pylab.bar(r1, pg_mos_mu, width = barWidth, color = 'blue', \
          edgecolor = 'black', yerr=pg_mos_var, capsize=7, \
          label='Pair-GAN')

pylab.bar(r2, cg_mos_mu, width = barWidth, color = 'cyan', \
        edgecolor = 'black', yerr=cg_mos_var, capsize=7, label='Cycle-GAN')

pylab.bar(r3, ref_mos_mu, width = barWidth, color = 'yellow', \
        edgecolor = 'black', yerr=ref_mos_var, capsize=7, label='Reference')

pylab.xticks([r + barWidth for r in range(len(pg_mos_mu))], \
              ['Neu-Ang', 'Neu-Hap', 'Neu-Sad'])
pylab.ylabel('MOS Score')
pylab.legend(loc=2), pylab.title('VESUS MOS Score')

#pylab.figure()
#pylab.subplot(121), pylab.boxplot([enc_dec_emo, lstm_emo, cycle_gan_emo], \
#             labels=['Enc-Dec-Gen', 'Bi-LSTM', 'Cycle-GAN'])
#pylab.title("Emotional Saliency ({})".format(emo_pair))
#pylab.subplot(122), pylab.boxplot([enc_dec_mos, lstm_mos, cycle_gan_mos], \
#             labels=['Enc-Dec-Gen', 'Bi-LSTM', 'Cycle-GAN'])
#pylab.title("MOS ({})".format(emo_pair))
#pylab.savefig('/home/ravi/Desktop/pitch-lddmm-spect/test_crowd_sourcing/'+emo_pair+'.png')
#pylab.close()

