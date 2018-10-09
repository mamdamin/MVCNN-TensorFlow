#!/bin/bash

#./setupTensorBoard.sh &

for i in `seq -w $1`
do
         qsub -q LUNG@argon-hm-p40-compute-8-24 -pe 56cpn 56 -o ~/TF_MVCNN.log -e TF_MVCNNError.log ~/MVCNN-TensorFlow-master/MVCNN_Airway_train
done

#  -p -512 priority