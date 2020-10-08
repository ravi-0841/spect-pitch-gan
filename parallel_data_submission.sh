#!/bin/bash

emo="cmu-arctic"
mode="train"
job_counter=1
for i in {1..2264..1}
do
    if [ -f "./data/${emo}/momenta/f0-train-${i}.mat" ]
    then
        echo "file ${i} exists"
    else
	    echo "Current job is $job_counter"
    	sbatch -J "${mode}_${i}" -o "./txt_files/${emo}_${mode}_${i}.txt" data_creation_job.sh $i $emo $mode
	    job_counter=$((job_counter+1))
    fi
done

#emo="neu-hap"
#mode="train"
#job_counter=1
#for i in {1..123..1}
#do
#    if [ -f "./data/${emo}/momenta/f0-train-${i}.mat" ]
#    then
#        echo "file ${i} exists"
#    else
#	    echo "Current job is $job_counter"
#    	sbatch -J "${mode}_${i}" -o "./txt_files/${emo}_${mode}_${i}.txt" data_creation_job.sh $i $emo $mode
#	    job_counter=$((job_counter+1))
#    fi
#done
#
#emo="neu-sad"
#mode="train"
#job_counter=1
#for i in {1..140..1}
#do
#    if [ -f "./data/${emo}/momenta/f0-train-${i}.mat" ]
#    then
#        echo "file ${i} exists"
#    else
#	    echo "Current job is $job_counter"
#    	sbatch -J "${mode}_${i}" -o "./txt_files/${emo}_${mode}_${i}.txt" data_creation_job.sh $i $emo $mode
#	    job_counter=$((job_counter+1))
#    fi
#done

#mode="valid"
#job_counter=1
#for i in {1..73..1}
#do
#    if [ -f "./data/${emo}/momenta/f0-valid-${i}.mat" ]
#    then
#        echo "file ${i} exists"
#    else
#	    echo "Current job is $job_counter"
#	    sbatch -J "${mode}_${i}" -o "./txt_files/${emo}_${mode}_${i}.txt" data_creation_job.sh $i $emo $mode
#	    job_counter=$((job_counter+1))
#    fi
#done
#
#
#mode="test"
#job_counter=1
#for i in {1..65..1}
#do
#    if [ -f "./data/${emo}/momenta/f0-test-${i}.mat" ]
#    then
#        echo "file ${i} exists"
#    else
#	    echo "Current job is $job_counter"
#	    sbatch -J "${mode}_${i}" -o "./txt_files/${emo}_${mode}_${i}.txt" data_creation_job.sh $i $emo $mode
#	    job_counter=$((job_counter+1))
#    fi
#done


# emo="neu-hap"
# kx=6
# mode="train"
# job_counter=1
# for i in {1..25000..25000}
# do
# 	end=$((i+25000-1))
# 	echo "Current start is $i end is $end and job is $job_counter"
# #	~/MATLAB/bin/matlab -nodesktop -nosplash -r "generate_vertical_data $i $end $job_counter $mode"
# 	# if [ $job_counter -eq 11 ]; then
# 	sbatch -o "NH_${mode}_${job_counter}.txt" data_job.sh $i $end $emo $job_counter $mode $kx
# 	# fi
# 	job_counter=$((job_counter+1))
# done

# job_counter=2
# for i in {25001..51619..26619}
# do
# 	end=$((i+26619-1))
# 	echo "Current start is $i end is $end and job is $job_counter"
# #	~/MATLAB/bin/matlab -nodesktop -nosplash -r "generate_vertical_data $i $end $job_counter $mode"
# 	# if [ $job_counter -eq 11 ]; then
# 	sbatch -o "NH_${mode}_${job_counter}.txt" data_job.sh $i $end $emo $job_counter $mode $kx
# 	# fi
# 	job_counter=$((job_counter+1))
# done


# mode="valid"
# job_counter=1
# for i in {1..3092..3092}
# do
# 	end=$((i+3092-1))
# 	echo "Current start is $i end is $end and job is $job_counter"
# 	# ~/MATLAB/bin/matlab -nodesktop -nosplash -r "generate_vertical_data $i $end $job_counter $mode"
# 	# if [ $job_counter -eq 1 ]; then
# 	sbatch -o "NH_${mode}_${job_counter}.txt" data_job.sh $i $end $emo $job_counter $mode $kx
# 	# fi
# 	job_counter=$((job_counter+1))
# done


# mode="test"
# job_counter=1
# for i in {1..2731..2731}
# do
# 	end=$((i+2731-1))
# 	echo "Current start is $i end is $end and job is $job_counter"
# 	# ~/MATLAB/bin/matlab -nodesktop -nosplash -r "generate_vertical_data $i $end $job_counter $mode"
# 	# if [ $job_counter -eq 1 ]; then
# 	sbatch -o "NH_${mode}_${job_counter}.txt" data_job.sh $i $end $emo $job_counter $mode $kx
# 	# fi
# 	job_counter=$((job_counter+1))
# done



# emo="neu-sad"
# kx=6

# mode="train"
# job_counter=1
# integerarray=(1000 1012 1026 1029 1032 1036 1037 1038 1041 1042 1061 1145 1147 1157 1159 1170 1176 1181 1192 1196 1207 1209 1212 1216 1219 1230 1231 1232 1233 1234 1235 1236 1237 1238 1239 1243 1244 1249 1252 1256 1258 1264 1265 1273 1274 1297 1298 1307 1312 1314 1318 1324 1325 1326 1328 1331 1336 1337 1357 1358 1362 1363 1370 1371 1375 1381 1382 1383 1387 1392 1393 1394 1397 1403 1404 1406 1408 1409 1410 1419 1434 835 837 889 934 939 946 957 958 960 964 965 967 981 993) # ${integerarray[@]}
# #for i in {1..1437..1}
# for i in ${integerarray[@]}
# do
# 	end=$((i+1-1))
# 	echo "Current start is $i end is $end and job is $job_counter"
# 	# if [ $job_counter -eq 9 -o $job_counter -eq 13 ]; then
# 	sbatch -o "./txt_files/NS_${mode}_${job_counter}.txt" data_creation_job.sh $i $end $emo $i $mode $kx #replace second $i with $job_counter
# 	# fi
# 	job_counter=$((job_counter+1))
# done


# mode="valid"
# job_counter=1
# integerarray=(10 14 21 23 29 30 31 32 37 55 68 69 7 71 8) # ${integerarray[@]}
# #for i in {1..75..1}
# for i in ${integerarray[@]}
# do
# 	end=$((i+1-1))
# 	echo "Current start is $i end is $end and job is $job_counter"
# 	# if [ $job_counter -eq 1 ]; then
# 	sbatch -o "./txt_files/NS_${mode}_${job_counter}.txt" data_creation_job.sh $i $end $emo $i $mode $kx
# 	# fi
# 	job_counter=$((job_counter+1))
# done


# mode="test"
# job_counter=1
# integerarray=(10 12 16 19 22 26 28 29 30 33 35 48 49 5 56 60 7 70 8 9) # ${integerarray[@]}
# #for i in {1..70..1}
# for i in ${integerarray[@]}
# do
# 	end=$((i+1-1))
# 	echo "Current start is $i end is $end and job is $job_counter"
# 	# if [ $job_counter -eq 1 ]; then
# 	sbatch -o "./txt_files/NS_${mode}_${job_counter}.txt" data_creation_job.sh $i $end $emo $i $mode $kx
# 	# fi
# 	job_counter=$((job_counter+1))
# done


# NA - train (1534), valid (72), test (61)
# NS - train (1437), valid (75), test (70)
