#! /bin/bash

#cycle_array_pitch=( 1e-06 0.00001 0.0001 0.01 )
#cycle_array_energy=( 1e-06 0.00001 0.001 0.1 )
#
#counter=1
#for p in "${cycle_array_pitch[@]}"
#do
#    for e in "${cycle_array_energy[@]}"
#    do
#        echo $counter;
#        sbatch -J $counter -o "./txt_files/mdw_${counter}.txt" gen_disc_job_energy_f0_momenta_discriminate_wasserstein.sh $p $e
#        counter=$((counter+1))
#    done
#done


sbatch -J job1 -o "./txt_files/NA_1.txt" gen_disc_job_energy_f0_momenta_discriminate_wasserstein.sh 1e-05 0.001 neu-ang
sbatch -J job2 -o "./txt_files/NA_2.txt" gen_disc_job_energy_f0_momenta_discriminate_wasserstein.sh 0.0001 0.001 neu-ang
sbatch -J job3 -o "./txt_files/NA_3.txt" gen_disc_job_energy_f0_momenta_discriminate_wasserstein.sh 1e-05 0.1 neu-ang
sbatch -J job4 -o "./txt_files/NA_4.txt" gen_disc_job_energy_f0_momenta_discriminate_wasserstein.sh 1e-06 1e-05 neu-ang

