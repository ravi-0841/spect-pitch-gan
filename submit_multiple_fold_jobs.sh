#! /bin/bash

# Running for different folds
counter=1
for f in {1..5..1}
do
    for c in {1..4..1}
    do
        echo $counter;
        sbatch -J $counter -o "./txt_files/NA_fold_${f}_count_${c}.txt" gen_disc_job_sum_ec_f0_fold.sh 1e-05 0.1 21 neu-ang $f $c
        sleep 5s
        counter=$((counter+1))
    done
done


# NA - lp 1e-05 le 0.1 li 0.0 seed 21
# NH - lp 0.0001 le 0.001 li 0.0 seed 4
# NS - lp 0.0001 le 0.1 li 0.0 seed 11
