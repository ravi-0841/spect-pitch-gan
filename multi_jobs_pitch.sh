#! /bin/bash

#cycle_array=( 1e-6 0.00001 0.0001 0.001 0.1 )
#momenta_array=( 0 1e-6 1e-5 1e-4 1e-3 1e-2 )
#
#counter=1
#for c in "${cycle_array[@]}"
#do
#    for m in "${momenta_array[@]}"
#    do
#        sbatch -J "job_${counter}" -o "./txt_files/job_${counter}.txt" gen_disc_job_lvi_mixed.sh $c $m
#        counter=$((counter+1))
#    done
#done

sbatch -J "job" -o "./txt_files/jobNH_1.txt" gen_disc_job_lvi.sh 0.0001 0.01
sbatch -J "job" -o "./txt_files/jobNH_2.txt" gen_disc_job_lvi.sh 0.0001 1e-05
sbatch -J "job" -o "./txt_files/jobNH_3.txt" gen_disc_job_lvi.sh 0.001 0.001
sbatch -J "job" -o "./txt_files/jobNH_4.txt" gen_disc_job_lvi.sh 0.001 1e-06
sbatch -J "job" -o "./txt_files/jobNH_5.txt" gen_disc_job_lvi.sh 1e-05 0.0001
sbatch -J "job" -o "./txt_files/jobNH_6.txt" gen_disc_job_lvi.sh 1e-05 1e-06
