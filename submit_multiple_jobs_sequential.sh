#! /bin/bash

cycle_array_mfc=( 0.001 0.01 0.1 )
predictor_learning_rate=( 0.00001 0.0001 0.001 )
discriminator_learning_rate=( 0.00001 0.0001 0.001 )
counter=1
for c in "${cycle_array_mfc[@]}"
do
    for p in "${predictor_learning_rate[@]}"
    do
        for d in "${discriminator_learning_rate[@]}"
        do
            if [ "$counter" -ge 1 ];
            then
                echo $counter;
                sbatch -J $counter -o "./txt_files/job_seq_${counter}.txt" gen_disc_job_separate_discriminate_sequential.sh $c $p $d
            fi;
            counter=$((counter+1))
        done
    done
done
