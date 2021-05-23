#!/bin/bash

for f in {1..5..1}
do
	for r in {1..4..1}
	do
		sbatch -J NA_$f_$r -o "./txt_files/NA_fold_${f}_run_${r}.txt" evaluate.sh neu-ang $f $r
	done
done

for f in {1..5..1}
do
	for r in {1..4..1}
	do
		sbatch -J NH_$f_$r -o "./txt_files/NH_fold_${f}_run_${r}.txt" evaluate.sh neu-hap $f $r
	done
done

for f in {1..5..1}
do
	for r in {1..4..1}
	do
		sbatch -J NS_$f_$r -o "./txt_files/NS_fold_${f}_run_${r}.txt" evaluate.sh neu-sad $f $r
	done
done
