#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 22:24:50 2020

@author: ravi
"""

import pylab
import joblib
import numpy as np
import scipy.stats as scistat


def get_p_values(emo_pair):
    data_dict       = joblib.load(emo_pair+'_vesus_results.pkl')
    
    print(emo_pair)
    print('GMM_emo: ', scistat.ttest_ind(data_dict['pg_emo'], 
                                         data_dict['gmm_emo']))
    print('LSTM_emo: ', scistat.ttest_ind(data_dict['pg_emo'], 
                                          data_dict['lstm_emo']))
    print('CGAN_emo: ', scistat.ttest_ind(data_dict['pg_emo'], 
                                          data_dict['cg_emo']))
    print('GMM_mos: ', scistat.ttest_ind(data_dict['pg_mos'], 
                                         data_dict['gmm_mos']))
    print('LSTM_mos: ', scistat.ttest_ind(data_dict['pg_mos'], 
                                          data_dict['lstm_mos']))
    print('CGAN_mos: ', scistat.ttest_ind(data_dict['pg_mos'], 
                                          data_dict['cg_mos']))
    
    
    print('\n\n')


if __name__=="__main__":
    
    emo_pairs       = ['neu-ang', 'neu-hap', 'neu-sad']
    
    edg_emo_mu      = list()
    edg_emo_std     = list()
    edg_mos_mu      = list()
    edg_mos_std     = list()
    
    lstm_emo_mu     = list()
    lstm_emo_std    = list()
    lstm_mos_mu     = list()
    lstm_mos_std    = list()
    
    cg_emo_mu       = list()
    cg_emo_std      = list()
    cg_mos_mu       = list()
    cg_mos_std      = list()
    
    gmm_emo_mu      = list()
    gmm_emo_std     = list()
    gmm_mos_mu      = list()
    gmm_mos_std     = list()
    
    
    for emo_pair in emo_pairs:
        
        get_p_values(emo_pair)
        data_dict       = joblib.load(emo_pair+'_vesus_results.pkl')
        
        edg_emo_mu.append(np.mean(data_dict['edg_emo']))
        edg_emo_std.append(np.var(data_dict['edg_emo']))
        edg_mos_mu.append(np.mean(data_dict['edg_mos']))
        edg_mos_std.append(np.var(data_dict['edg_mos']))
        
        lstm_emo_mu.append(np.mean(data_dict['lstm_emo']))
        lstm_emo_std.append(np.var(data_dict['lstm_emo']))
        lstm_mos_mu.append(np.mean(data_dict['lstm_mos']))
        lstm_mos_std.append(np.var(data_dict['lstm_mos']))
        
        cg_emo_mu.append(np.mean(data_dict['cg_emo']))
        cg_emo_std.append(np.var(data_dict['cg_emo']))
        cg_mos_mu.append(np.mean(data_dict['cg_mos']))
        cg_mos_std.append(np.var(data_dict['cg_mos']))
        
        gmm_emo_mu.append(np.mean(data_dict['gmm_emo']))
        gmm_emo_std.append(np.var(data_dict['gmm_emo']))
        gmm_mos_mu.append(np.mean(data_dict['gmm_mos']))
        gmm_mos_std.append(np.var(data_dict['gmm_mos']))
    
#    barWidth=0.2
#    r1 = np.arange(len(edg_emo_mu))
#    r2 = [x + barWidth for x in r1]
#    r3 = [x + 2*barWidth for x in r1]
#    r4 = [x + 3*barWidth for x in r1]
#     
#    # Emotional Saliency plot
#    pylab.figure()
#    pylab.bar(r1, edg_emo_mu, width = barWidth, color = 'blue', \
#              edgecolor = 'black', yerr=edg_emo_std, capsize=7, \
#              label='Enc-Dec-Gen')
#    
#    pylab.bar(r2, lstm_emo_mu, width = barWidth, color = 'cyan', \
#            edgecolor = 'black', yerr=lstm_emo_std, capsize=7, \
#            label='Bi-LSTM')
#    
#    pylab.bar(r3, cg_emo_mu, width = barWidth, color = 'orange', \
#            edgecolor = 'black', yerr=cg_emo_std, capsize=7, \
#            label='Cycle-GAN')
#    
#    pylab.bar(r4, gmm_emo_mu, width = barWidth, color = 'yellow', \
#            edgecolor = 'black', yerr=gmm_emo_std, capsize=7, \
#            label='GMM')
#    
#    pylab.xticks([r + 1.0*barWidth for r in range(len(edg_emo_mu))], \
#                  ['Neu-Ang', 'Neu-Hap', 'Neu-Sad'])
#    pylab.ylabel('Emotional Saliency')
#    pylab.legend(loc=1), pylab.title('VESUS Emotional Saliency')
#    
#    
#    # MOS score plot
#
#    pylab.figure()
#    pylab.bar(r1, edg_mos_mu, width = barWidth, color = 'blue', \
#              edgecolor = 'black', yerr=edg_mos_std, capsize=7, \
#              label='Enc-Dec-Gen')
#    
#    pylab.bar(r2, lstm_mos_mu, width = barWidth, color = 'cyan', \
#            edgecolor = 'black', yerr=lstm_mos_std, capsize=7, \
#            label='Bi-LSTM')
#    
#    pylab.bar(r3, cg_mos_mu, width = barWidth, color = 'orange', \
#            edgecolor = 'black', yerr=cg_mos_std, capsize=7, \
#            label='Cycle-GAN')
#    
#    pylab.bar(r4, gmm_mos_mu, width = barWidth, color = 'yellow', \
#            edgecolor = 'black', yerr=gmm_mos_std, capsize=7, \
#            label='GMM')
#    
#    pylab.xticks([r + 1.0*barWidth for r in range(len(edg_mos_mu))], \
#                  ['Neu-Ang', 'Neu-Hap', 'Neu-Sad'])
#    pylab.ylabel('MOS Score')
#    pylab.legend(loc=1), pylab.title('VESUS MOS Score')
