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


def get_tsne_projection(data_matrix, dim=2):
    tsne_object = TSNE(n_components=dim, verbose=1)
    data_matrix_embedded = tsne_object.fit_transform(data_matrix)
    return data_matrix_embedded


def plot_scatter(embed_matrix, num_samples, title):
    pylab.figure()
    pylab.plot(embed_matrix[:num_samples,0], 
               embed_matrix[:num_samples,1], 
               'r.', label='Source F0')
    pylab.plot(embed_matrix[num_samples:2*num_samples,0], 
               embed_matrix[num_samples:2*num_samples,1], 
               'g.', label='Generated F0')
    pylab.plot(embed_matrix[2*num_samples:3*num_samples,0], 
               embed_matrix[2*num_samples:3*num_samples,1], 
               'b.', label='Target F0')
    pylab.legend(loc=1)
    pylab.title(title)


if __name__ == '__main__':
    
    data = scio.loadmat('/home/ravi/Desktop/vanilla_variational_NA_f0s.mat')
    
    f0_A = data['f0_A']
    f0_B = data['f0_B']
    f0_A2B_vcg = data['f0_A2B_vcg']
    f0_A2B_cg = data['f0_A2B_cg']
    
    num_samples = f0_A.shape[0]
    labels = np.concatenate((np.zeros((num_samples,1)), 
                             np.ones((num_samples,1)), 
                             2*np.ones((num_samples,1))), axis=0)
    
    data_cg = np.concatenate((f0_A, f0_A2B_cg, f0_B), axis=0)
    data_vcg = np.concatenate((f0_A, f0_A2B_vcg, f0_B), axis=0)
    
    data_cg_embedded = get_tsne_projection(data_cg)
    data_vcg_embedded = get_tsne_projection(data_vcg)
    
    van_pdist = np.sum((data_cg_embedded[num_samples:2*num_samples] \
                        - data_cg_embedded[2*num_samples:3*num_samples])**2, axis=1)
    var_pdist = np.sum((data_vcg_embedded[num_samples:2*num_samples] \
                        - data_vcg_embedded[2*num_samples:3*num_samples])**2, axis=1)
    
    plot_scatter(data_cg_embedded, num_samples=num_samples, title='Vanilla')
    plot_scatter(data_vcg_embedded, num_samples=num_samples, title='Variational')