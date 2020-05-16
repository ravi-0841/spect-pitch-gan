#! /bin/bash

cycle_array_pitch=( 1e-06 1e-04 0.01 )
cycle_array_mfc=( 1e-05 0.001 0.01 )
momenta_array=( 1e-06 1e-05 1e-04 )

counter=1
for p in "${cycle_array_pitch[@]}"
do
    for m in "${cycle_array_mfc[@]}"
    do
        for mo in "${momenta_array[@]}"
        do
            sbatch -J $counter -o "./txt_files/job${counter}.txt" gen_disc_job.sh $p $m $mo
            counter=$((counter+1))
        done
    done
done
