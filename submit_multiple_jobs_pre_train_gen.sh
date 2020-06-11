#! /bin/bash

sbatch -J job_pt1 -o "./txt_files/job_pre_train_1.txt" gen_disc_job_pre_train_gen.sh 0.0001 1.0 1e-06 0.0001 0.01
sbatch -J job_pt2 -o "./txt_files/job_pre_train_2.txt" gen_disc_job_pre_train_gen.sh 0.0001 10.0 1e-06 0.0001 0.01
sbatch -J job_pt3 -o "./txt_files/job_pre_train_3.txt" gen_disc_job_pre_train_gen.sh 1e-05 1.0 1e-06 0.0001 0.01
sbatch -J job_pt4 -o "./txt_files/job_pre_train_4.txt" gen_disc_job_pre_train_gen.sh 1e-05 10.0 1e-06 0.0001 0.01
