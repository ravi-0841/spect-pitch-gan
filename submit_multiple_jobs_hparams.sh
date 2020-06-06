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



#generator_learning_rate=( 0.0000005 0.000001 0.000005 0.00001 0.00002 0.00005 0.0001 )
#discriminator_learning_rate=( 0.0001 0.0002 0.0005 0.001 0.003 0.007 0.01)
generator_learning_rate=( 0.000005 0.00001 0.0001 )
discriminator_learning_rate=( 0.0001 0.001 0.01 )
cycle_pitch=( 1e-05 1e-04 1 )
cycle_mfc=( 0.1 1 10 )

counter=1
for g in "${generator_learning_rate[@]}"
do
    for d in "${discriminator_learning_rate[@]}"
    do
        for cp in "${cycle_pitch[@]}"
        do
            for cm in "${cycle_mfc[@]}"
            do
                if [ "$counter" -ge 1 ];
                then
                    echo $counter;
                    sbatch -J $counter -o "./txt_files/job_lr_${counter}.txt" gen_disc_job_analyze_weights_hparams.sh $g $d $cp $cm
                fi;
                counter=$((counter+1))
            done
        done
    done
done
