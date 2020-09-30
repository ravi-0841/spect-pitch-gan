#! /bin/bash

cycle_array_pitch=( 1e-06 0.00001 0.0001 )
cycle_array_energy=( 0.00001 0.001 0.1 )

counter=1
for p in "${cycle_array_pitch[@]}"
do
    for e in "${cycle_array_energy[@]}"
    do
        echo $counter;
        sbatch -J $counter -o "./txt_files/mdw_${counter}.txt" gen_disc_job_energy_f0_momenta_discriminate_wasserstein.sh $p $e
        counter=$((counter+1))
    done
done
