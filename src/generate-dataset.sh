#!/bin/bash
#OAR -n "Dataset Generation"
#OAR -p gpu>0
#OAR -l /gpu=1,walltime=20:00:00
#OAR -O logs/generation.log
#OAR -E logs/generation-error.log
#OAR -t besteffort

export http_proxy=http://11.0.0.254:3142/
export https_proxy=http://11.0.0.254:3142/

cd ~/M2-Prestel-state-from-obs-ML
NSLOTS=$(cat $OAR_NODEFILE | wc -l)

echo $NSLOTS

source ./venv/bin/activate
ls -s /scratch-local/vforiel ./M2-Prestel-state-from-obs-ML/scratch-local

python Generate_A_Little_Bit_Physical_Dataset.py

exit 0