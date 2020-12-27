#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 14:41:10 2020

@author: ravi
"""

import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    ## files
    parser.add_argument('--data_folder', default='/home/ravi/Downloads/Emo-Conv/neutral-angry/all_above_0.5', 
                        type=str, help="location to read data from")
    parser.add_argument('--save_at', default="/home/ravi/Desktop/", type=str, 
                        help="location to save data")

    # extraction scheme
    parser.add_argument('--emo_pair', default='neutral_angry', type=str, 
                        help="emotion pair")
    parser.add_argument('--fold', default=1, type=int, 
                        help="choice of the fold choose from 1-5")
    parser.add_argument('--frame_len', default=128, type=int)
    parser.add_argument('--eval_size', default=0.03, type=float)

    parser.add_argument('--encode_raw_spect', default=False, type=bool, 
                        help="encode raw spectrum using mfcc of energies")
    parser.add_argument('--n_mels', default=23, type=float, 
                        help="#mfcc features")
    parser.add_argument('--n_samplings', default=10, type=int, 
                        help="# of samplings")
    parser.add_argument('--fs', default=16000, type=int, 
                        help="Sampling rate")
    parser.add_argument('--win_len', default=0.005, type=float, 
                        help="window size in sec")
    parser.add_argument('--hop_len', default=0.005, type=float, 
                        help="window stride in sec")
