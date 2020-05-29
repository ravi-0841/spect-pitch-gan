#! /bin/bash

cycle_array_pitch=( 0.1 1 5 )
cycle_array_mfc=( 1 10 20 )
momenta_array=( 1e-06 1e-04 )

counter=1
for p in "${cycle_array_pitch[@]}"
do
    for m in "${cycle_array_mfc[@]}"
    do
        for mo in "${momenta_array[@]}"
        do
            if [ "$counter" -ge 1 ];
            then
                echo $counter;
                sbatch -J $counter -o "./txt_files/job${counter}.txt" gen_disc_job_separate_discriminate.sh $p $m $mo
            fi;
            counter=$((counter+1))
        done
    done
done
