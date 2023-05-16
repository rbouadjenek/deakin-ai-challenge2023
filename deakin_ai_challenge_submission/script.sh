#!/bin/bash

cd program/
# CONDA
conda create -n mypython3 python=3.11 
source activate mypython3 
# conda install numpy 
conda install tensorflow 
# conda install keras 
# conda install pillow 
# conda install h5py 

python3 deakin_ai_challenge_submission.py $1 $2 
