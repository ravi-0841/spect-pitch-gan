import pylab
import os
import re
import argparse
import numpy as np

from glob import glob

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Arguments for plotting losses")

    parser.add_argument("--dir", type=str, \
            help="directory containing logs", default="./log/")
    
    args = parser.parse_args()
    
    files = sorted(glob(os.path.join(args.dir, '*.log')))

    for f in files:

        print(f)
        name = os.path.basename(f)[:-4]#f.split('/')[-1]
        train_generator_loss = list()
        train_discriminator_loss = list()

        train_momenta_A2B_loss = list()
        train_momenta_B2A_loss = list()
        train_pitch_A2B_loss = list()
        train_pitch_B2A_loss = list()
        train_mfc_A2B_loss = list()
        train_mfc_B2A_loss = list()

        r = open(f,'r')
        for line in r:
            try:
                if "Train Generator" in line:
                    n = re.findall(r"[-+]?\d*\.\d+e-\d+|[-+]?\d*\.\d+|\d+", line)
                    train_generator_loss.append(-0.8*float(n[0]))
                if "Train Discriminator" in line:
                    n = re.findall(r"[-+]?\d*\.\d+e-\d+|[-+]?\d*\.\d+|\d+", line)
                    train_discriminator_loss.append(-0.9*float(n[0]))
                if "Train Momenta A2B" in line:
                    n = re.findall(r"[-+]?\d*\.\d+e-\d+|[-+]?\d*\.\d+|\d+", line)
                    train_momenta_A2B_loss.append(float(n[1]))
                if "Train Momenta B2A" in line:
                    n = re.findall(r"[-+]?\d*\.\d+e-\d+|[-+]?\d*\.\d+|\d+", line)
                    train_momenta_B2A_loss.append(float(n[1]))
                if "Train Pitch A2B" in line:
                    n = re.findall(r"[-+]?\d*\.\d+e-\d+|[-+]?\d*\.\d+|\d+", line)
                    train_pitch_A2B_loss.append(float(n[1]))
                if "Train Pitch B2A" in line:
                    n = re.findall(r"[-+]?\d*\.\d+e-\d+|[-+]?\d*\.\d+|\d+", line)
                    train_pitch_B2A_loss.append(float(n[1]))
                if "Train Mfc A2B" in line:
                    n = re.findall(r"[-+]?\d*\.\d+e-\d+|[-+]?\d*\.\d+|\d+", line)
                    train_mfc_A2B_loss.append(float(n[1]))
                if "Train Mfc B2A" in line:
                    n = re.findall(r"[-+]?\d*\.\d+e-\d+|[-+]?\d*\.\d+|\d+", line)
                    train_mfc_B2A_loss.append(float(n[1]))

            except Exception as e:
                print(e)

        diff = np.abs(np.asarray(train_generator_loss) - np.asarray(train_discriminator_loss))
        r.close()
        if len(train_generator_loss)>0:
            pylab.figure(figsize=(12,12))
            pylab.plot(train_generator_loss, 'r', label="generator loss", linewidth=3)
            pylab.plot(train_discriminator_loss, 'g', label="discriminator loss", linewidth=3)
            pylab.grid(), pylab.legend(loc=1, fontsize=15)
            pylab.suptitle(name)
            pylab.savefig(args.dir+name+'.png')
            pylab.close()
        elif len(train_momenta_A2B_loss)>0:
            pylab.figure(figsize=(12,12))
            pylab.subplot(131)
            pylab.plot(train_momenta_A2B_loss, 'r', label="Momenta A2B loss")
            pylab.plot(train_momenta_B2A_loss, 'g', label="Momenta B2A loss")
            pylab.grid(), pylab.legend(loc=1)
            pylab.subplot(132)
            pylab.plot(train_pitch_A2B_loss, 'r', label="Pitch A2B loss")
            pylab.plot(train_pitch_B2A_loss, 'g', label="Pitch B2A loss")
            pylab.grid(), pylab.legend(loc=1)
            pylab.subplot(133)
            pylab.plot(train_mfc_A2B_loss, 'r', label="Mfc A2B loss")
            pylab.plot(train_mfc_B2A_loss, 'g', label="Mfc B2A loss")
            pylab.grid(), pylab.legend(loc=1)
            pylab.suptitle(name)
            pylab.savefig(args.dir+name+'.png')
            pylab.close()
