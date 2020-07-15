#! /bin/bash

cycle_array_pitch=( 0.001 0.01 )
cycle_array_mfc=( 0.01 0.1 1 )
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
<<<<<<< Updated upstream
                sbatch -J $counter -o "./txt_files/job${counter}.txt" gen_disc_job_mfc_normalize.sh $p $m $mo
=======
                sbatch -J $counter -o "./txt_files/job${counter}.txt" gen_disc_job_label_flipped.sh $p $m $mo
>>>>>>> Stashed changes
            fi;
            counter=$((counter+1))
        done
    done
done
