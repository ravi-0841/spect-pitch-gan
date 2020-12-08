#! /bin/bash

for f in ./log/sum_mfc_wstn_neu-hap_random_seed/*.log; 
do
    echo $f;
    tail -n5 $f;
done
