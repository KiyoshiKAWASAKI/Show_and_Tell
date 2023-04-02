#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -l h=!qa-a10-005&!qa-rtx6k-044&!qa-a10-006
#$ -e errors/
#$ -N generate_full_icwsm

# Required modules
module load conda
conda init bash
source activate show_and_tell

python3 main.py --phase=test \
                --model_file='/afs/crc.nd.edu/group/cvrl/scratch_49/jhuang24/safe_data/show_and_tell_data/39999.npy'