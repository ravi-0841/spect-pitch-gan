#! /bin/bash

for f in ./log/sum_mfc_wstn_neu-sad/*.log; 
do
    echo $f;
    tail -n5 $f;
done
