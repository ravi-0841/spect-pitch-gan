#! /bin/bash

#cycle_array_pitch=( 0.0001 0.001 0.01 )
#cycle_array_mfc=( 0.01 0.1 1 )
#momenta_array=( 1e-06 1e-04 0.001 )
#
#counter=1
#for p in "${cycle_array_pitch[@]}"
#do
#    for m in "${cycle_array_mfc[@]}"
#    do
#        for mo in "${momenta_array[@]}"
#        do
#            if [ "$counter" -ge 1 ];
#            then
#                echo $counter;
#                sbatch -J $counter -o "./txt_files/job${counter}.txt" gen_disc_job_analyze_weights.sh $p $m $mo
#            fi;
#            counter=$((counter+1))
#        done
#    done
#done



generator_learning_rate=( 0.0000001 0.00001 0.001 0.01 )
discriminator_learning_rate=( 0.0000001 0.00001 0.001 0.01)

counter=1
for g in "${generator_learning_rate[@]}"
do
    for d in "${discriminator_learning_rate[@]}"
    do
        if [ "$counter" -ge 1 ];
        then
            echo $counter;
            sbatch -J $counter -o "./txt_files/job_lr_${counter}.txt" gen_disc_job_analyze_weights_lr.sh $g $d
        fi;
        counter=$((counter+1))
    done
done
