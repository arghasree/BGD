#!/bin/bash
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00-06:00      # time (DD-HH:MM)
#SBATCH --account=def-guzdial 

echo "transfer start"
date
rsync -av --exclude='models/AlexNet' --exclude='camelyon17' ~/scratch/XAI/ $SLURM_TMPDIR/XAI/

echo "transfer done"
date # Print for timing, compare with above


cd $SLURM_TMPDIR/XAI

echo "load"
module load python/3.10 cuda cudnn

# create and activate virtual environment
echo "create env"
virtualenv --no-download  $SLURM_TMPDIR/pytorch_env
source $SLURM_TMPDIR/pytorch_env/bin/activate


# install python packages
pip install --upgrade pip
pip install --upgrade pandas
pip install --upgrade kagglehub
pip install --upgrade torch
pip install --upgrade torchvision
#pip install --no-index -r requirements.txt
pip install --upgrade sklearn
pip install --upgrade matplotlib
pip install --upgrade numpy
pip install --upgrade scipy
pip install --upgrade kagglehub
pip install --upgrade shutil
pip install --upgrade PIL

pip list
python main.py \
--dataset 'MNIST' \
--epochs 10 \
--imbalanced 'False' \
--model_type MLP \
--reloaded 'False' \
--copy_weight_reinit 'True' \
--random_seed 42 \
--num_classes 10 \
--directory '/home/arghasre/scratch/XAI/' 

echo "finish"
