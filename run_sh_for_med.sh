#!/bin/bash
echo "start"
python main.py --dataset MED \
--imbalanced 'False' \
--model_type AlexNet \
--batch_size 128 \
--learning_rate 0.01 \
--weight_decay 0.0001 \
--lr_decay 0.1 \
--random_seed 42 \
--num_classes 10 \
--directory ./
