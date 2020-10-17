#! /bin/bash

cycle_array_pitch=( 1e-06 0.00001 0.0001 0.01 )
cycle_array_energy=( 1e-06 0.00001 0.001 0.1 )

counter=1
for p in "${cycle_array_pitch[@]}"
do
    for e in "${cycle_array_energy[@]}"
    do
        echo $counter;
        sbatch -J $counter -o "./txt_files/sum_ec_mdw_${counter}.txt" gen_disc_job_energy_f0_momenta_discriminate_wasserstein.sh $p $e neu-ang
        counter=$((counter+1))
    done
done


#sbatch -J NA1 -o "./txt_files/NA_1.txt" gen_disc_job_energy_f0_momenta_discriminate_wasserstein.sh 1e-05 0.001 neu-ang
#sbatch -J NA2 -o "./txt_files/NA_2.txt" gen_disc_job_energy_f0_momenta_discriminate_wasserstein.sh 0.0001 0.001 neu-ang
#sbatch -J NA3 -o "./txt_files/NA_3.txt" gen_disc_job_energy_f0_momenta_discriminate_wasserstein.sh 1e-05 0.1 neu-ang
#sbatch -J NA4 -o "./txt_files/NA_4.txt" gen_disc_job_energy_f0_momenta_discriminate_wasserstein.sh 1e-06 1e-05 neu-ang
#
#sbatch -J NH1 -o "./txt_files/NH_1.txt" gen_disc_job_energy_f0_momenta_discriminate_wasserstein.sh 1e-05 0.001 neu-hap
#sbatch -J NH2 -o "./txt_files/NH_2.txt" gen_disc_job_energy_f0_momenta_discriminate_wasserstein.sh 0.0001 0.001 neu-hap
#sbatch -J NH3 -o "./txt_files/NH_3.txt" gen_disc_job_energy_f0_momenta_discriminate_wasserstein.sh 1e-05 0.1 neu-hap
#sbatch -J NH4 -o "./txt_files/NH_4.txt" gen_disc_job_energy_f0_momenta_discriminate_wasserstein.sh 1e-06 1e-05 neu-hap
#
#sbatch -J NS1 -o "./txt_files/NS_1.txt" gen_disc_job_energy_f0_momenta_discriminate_wasserstein.sh 1e-05 0.001 neu-sad
#sbatch -J NS2 -o "./txt_files/NS_2.txt" gen_disc_job_energy_f0_momenta_discriminate_wasserstein.sh 0.0001 0.001 neu-sad
#sbatch -J NS3 -o "./txt_files/NS_3.txt" gen_disc_job_energy_f0_momenta_discriminate_wasserstein.sh 1e-05 0.1 neu-sad
#sbatch -J NS4 -o "./txt_files/NS_4.txt" gen_disc_job_energy_f0_momenta_discriminate_wasserstein.sh 1e-06 1e-05 neu-sad
