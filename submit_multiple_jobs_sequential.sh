#! /bin/bash

cycle_array_mfc=( 0.001 0.01 )
momenta_array=( 1e-06 1e-04 )

counter=1
for m in "${cycle_array_mfc[@]}"
do
    for mo in "${momenta_array[@]}"
    do
        if [ "$counter" -ge 1 ];
        then
            echo $counter;
            sbatch -J $counter -o "./txt_files/job_seq_${counter}.txt" gen_disc_job_separate_discriminate_sequential.sh $m $mo
        fi;
        counter=$((counter+1))
    done
done
