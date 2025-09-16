python3 masks/mask_main.py --dataset MNIST --model_type MLP --mask_initial_type uniform --lambda_reg 0.1 --mask_lr 1 --epochs 100
python3 masks/mask_main.py --dataset MNIST --model_type MLP --mask_initial_type uniform --lambda_reg 0.01 --mask_lr 1 --epochs 100
python3 masks/mask_main.py --dataset MNIST --model_type MLP --mask_initial_type uniform --lambda_reg 0.001 --mask_lr 1 --epochs 100
python3 masks/mask_main.py --dataset MNIST --model_type MLP --mask_initial_type zeros --lambda_reg 0.1 --mask_lr .1 --epochs 100
python3 masks/mask_main.py --dataset MNIST --model_type MLP --mask_initial_type ones --lambda_reg 0.1 --mask_lr .01 --epochs 100

