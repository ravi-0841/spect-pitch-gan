#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 21:33:56 2021

@author: ravi
"""

import scipy.io as scio
import numpy as np
import pylab
from sklearn.manifold import TSNE
from matplotlib import rc
import scipy.stats as stats


def get_tsne_projection(data_matrix, dim=2):
    tsne_object = TSNE(n_components=dim, verbose=1)
    data_matrix_embedded = tsne_object.fit_transform(data_matrix)
    return data_matrix_embedded


def plot_scatter(embed_matrix, num_samples, title):

    rc('font', weight='bold')
    rc('axes', linewidth=2)

    pylab.figure(figsize=(8,7))
    pylab.plot(embed_matrix[:num_samples,0], 
               embed_matrix[:num_samples,1], 
               'r.', label='Source F0')
    pylab.plot(embed_matrix[num_samples:2*num_samples,0], 
               embed_matrix[num_samples:2*num_samples,1], 
               'g.', label='Generated F0')
    pylab.plot(embed_matrix[2*num_samples:3*num_samples,0], 
               embed_matrix[2*num_samples:3*num_samples,1], 
               'b.', label='Target F0')
    
    pylab.xticks(size=14), pylab.yticks(size=14)
    pylab.grid(), pylab.legend(loc=1, prop={'size':13, 'weight':'bold'})
    pylab.title(title, fontsize=14, fontweight='bold')


def plot_bars_f0():
    pylab.rcParams['font.size'] = 28
    
    barWidth=0.2
    r1 = np.arange(1)
    r2 = [x + barWidth for x in r1]
    
    r3 = [x + 2*(barWidth)+0.15 for x in r1]
    r4 = [x + barWidth for x in r3]
    
    r5 = [x + 4*(barWidth)+2*0.15 for x in r1]
    r6 = [x + barWidth for x in r5]

    pylab.figure(figsize=(13, 9))
    pylab.bar(r1, 74.13, width = barWidth, 
              color = 'white', edgecolor = 'black', hatch = "////", 
              yerr=34.9, capsize=7, label='Vanilla CycleGAN')
    pylab.bar(r2, 67.16, width = barWidth, 
              color = 'white', edgecolor = 'black', hatch = "+++", 
              yerr=36.3, capsize=7, label='Variational CycleGAN')
    pylab.legend(loc=1, prop={'size':13, 'weight':'bold'})
    
    pylab.bar(r3, 63.67, width = barWidth, 
              color = 'white', edgecolor = 'black', hatch = "////", 
              yerr=28.6, capsize=7, label='Vanilla CycleGAN')
    pylab.bar(r4, 52.2, width = barWidth, 
              color = 'white', edgecolor = 'black', hatch = "+++", 
              yerr=27.4, capsize=7, label='Variational CycleGAN')

    pylab.bar(r5, 82.25, width = barWidth, 
              color = 'white', edgecolor = 'black', hatch = "////", 
              yerr=35.4, capsize=7, label='Vanilla CycleGAN')
    pylab.bar(r6, 57.3, width = barWidth, 
              color = 'white', edgecolor = 'black', hatch = "+++", 
              yerr=33.1, capsize=7, label='Variational CycleGAN')
    
    pylab.xticks([r for r in [r2[0]-barWidth/2, r4[0]-barWidth/2, r6[0]-barWidth/2]], \
                  ['Neutral-Angry', 'Neutral-Happy', 'Neutral-Sad'])
    pylab.ylabel('F0 Prediction Error (Hz)')


def plot_bars_ec():
    pylab.rcParams['font.size'] = 28
    rc('font', weight='bold')
    rc('axes', linewidth=2)
    
    barWidth=0.2
    r1 = np.arange(1)
    r2 = [x + barWidth for x in r1]
    
    r3 = [x + 2*(barWidth)+0.15 for x in r1]
    r4 = [x + barWidth for x in r3]
    
    r5 = [x + 4*(barWidth)+2*0.15 for x in r1]
    r6 = [x + barWidth for x in r5]

    pylab.figure(figsize=(13, 9))
    pylab.bar(r1, 49.05, width = barWidth, 
              color = 'white', edgecolor = 'black', hatch = "////", 
              yerr=17.0, capsize=7, label='Vanilla CycleGAN')
    pylab.bar(r2, 47.9, width = barWidth, 
              color = 'white', edgecolor = 'black', hatch = "++++", 
              yerr=17.0, capsize=7, label='Variational CycleGAN')
    pylab.legend(loc=2, prop={'size':18, 'weight':'bold'})
    
    pylab.bar(r3, 48.7, width = barWidth, 
              color = 'white', edgecolor = 'black', hatch = "////", 
              yerr=16.0, capsize=7, label='Vanilla CycleGAN')
    pylab.bar(r4, 47.5, width = barWidth, 
              color = 'white', edgecolor = 'black', hatch = "++++", 
              yerr=11.6, capsize=7, label='Variational CycleGAN')

    pylab.bar(r5, 48.3, width = barWidth, 
              color = 'white', edgecolor = 'black', hatch = "////", 
              yerr=16.0, capsize=7, label='Vanilla CycleGAN')
    pylab.bar(r6, 46.3, width = barWidth, 
              color = 'white', edgecolor = 'black', hatch = "++++", 
              yerr=13.6, capsize=7, label='Variational CycleGAN')
    
    pylab.xticks([r for r in [r2[0]-barWidth/2, r4[0]-barWidth/2, r6[0]-barWidth/2]], \
                  ['Neutral-Angry', 'Neutral-Happy', 'Neutral-Sad'])
    pylab.ylabel('Energy Prediction Error')


plot_bars_ec()

#if __name__ == '__main__':
#    
#    data = scio.loadmat('/home/ravi/Desktop/vanilla_variational_NA_f0s.mat')
#    
#    f0_A = data['f0_A']
#    f0_B = data['f0_B']
#    f0_A2B_vcg = data['f0_A2B_vcg']
#    f0_A2B_cg = data['f0_A2B_cg']
#    
#    num_samples = f0_A.shape[0]
#    labels = np.concatenate((np.zeros((num_samples,1)), 
#                             np.ones((num_samples,1)), 
#                             2*np.ones((num_samples,1))), axis=0)
    
    #get tsne-embeddings
#    data_cg = np.concatenate((f0_A, f0_A2B_cg, f0_B), axis=0)
#    data_vcg = np.concatenate((f0_A, f0_A2B_vcg, f0_B), axis=0)
#    
#    data_cg_embedded = get_tsne_projection(data_cg)
#    data_vcg_embedded = get_tsne_projection(data_vcg)
#    
#    van_pdist = np.sum((data_cg_embedded[num_samples:2*num_samples] \
#                        - data_cg_embedded[2*num_samples:3*num_samples])**2, axis=1) ** 0.5
#    var_pdist = np.sum((data_vcg_embedded[num_samples:2*num_samples] \
#                        - data_vcg_embedded[2*num_samples:3*num_samples])**2, axis=1) ** 0.5
    
    #perform one sample t-test
#    print("p-value: {}".format(stats.ttest_1samp(a=(var_pdist-van_pdist), popmean=15)))
#
#    print("MSE Loss (Vanilla): {}, {}".format(str(np.mean(van_pdist)), str(np.std(van_pdist))))
#    print("MSE Loss (Variational): {}, {}".format(str(np.mean(var_pdist)), str(np.std(var_pdist))))

#    plot_scatter(data_cg_embedded, num_samples=num_samples, title='Neutral-Sad (Vanilla CycleGAN)')
#    plot_scatter(data_vcg_embedded, num_samples=num_samples, title='Neutral-Sad (Variational CycleGAN)')
    
    #plot the generated F0 contours
#    rc('font', weight='bold')
#    rc('axes', linewidth=2)
#    r = np.random.permutation(np.arange(0, f0_A.shape[0], 1))
#    for i in range(100):
#    q = 382 #r[i]
#    pylab.figure(figsize=(11,9))
#    pylab.plot(f0_A[q], linewidth=2, label='Source F0')
#    pylab.plot(f0_A2B_cg[q], linewidth=2, label='Vanilla F0')
#    pylab.plot(f0_A2B_vcg[q], linewidth=2, label='Variational F0')
#    pylab.plot(f0_B[q], linewidth=2, label='Target F0')
#    pylab.xticks(size=20), pylab.yticks(size=20)
#    pylab.legend(loc=2, prop={'size':17, 'weight':'bold'})
#    pylab.title(str(q))
#    pylab.savefig('/home/ravi/Desktop/F0_comparisons/{}.png'.format(str(q)))
#    pylab.close()
    
