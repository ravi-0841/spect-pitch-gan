import pylab
import os
import re
import argparse

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

        diff = list()

        r = open(f,'r')
        for line in r:
            try:
                if "Train Generator" in line:
                    n = re.findall(r"[-+]?\d*\.\d+e-\d+|[-+]?\d*\.\d+|\d+", line)
                    train_generator_loss.append(float(n[0]))
                if "Train Discriminator" in line:
                    n = re.findall(r"[-+]?\d*\.\d+e-\d+|[-+]?\d*\.\d+|\d+", line)
                    train_discriminator_loss.append(float(n[0]))
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

                diff.append(np.abs(train_generator_loss[-1] - train_discriminator_loss[-1]))
            except Exception as e:
                pass

        r.close()
        if len(train_generator_loss)>0:
            pylab.figure(figsize=(12,12))
            pylab.subplot(121)
            pylab.plot(train_generator_loss, 'r', label="generator loss")
            pylab.plot(train_discriminator_loss, 'g', label="discriminator loss")
            pylab.grid(), pylab.legend(loc=1)
            pylab.title(name)
            pylab.subplot(122)
            pylab.plot(diff, 'g', label='Difference loss')
            pylab.title('Difference')
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
