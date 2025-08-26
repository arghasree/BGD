python main.py --dataset 'MED' --num_classes 2 --imbalanced 'False' --epochs 50 --model_type 'AlexNet' --batch_size 128 --learning_rate 0.01 --weight_decay 0.0001 --lr_decay 0.1 --random_seed 21 --layer_name 'conv2.weight' --directory '/home/arghasre/scratch/XAI/' # have to mention this for saving models etc.

python main.py --dataset 'MED' --num_classes 2 --imbalanced 'False' --epochs 50 --model_type 'AlexNet' --batch_size 128 --learning_rate 0.01 --weight_decay 0.0001 --lr_decay 0.1 --random_seed 21 --directory '/home/arghasre/scratch/XAI/' # have to mention this for saving models etc.

git add . ':!camelyon17/' ':!models/AlexNet/' ':!models/MLP/' ':!slurm*'
git commit -m "Fir commit"
git push