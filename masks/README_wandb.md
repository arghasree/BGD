# Weights & Biases Integration for Mask Regularization

This document explains how to use Weights & Biases (wandb) to track your mask regularization experiments.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python setup_wandb.py
```

### 2. Login to Wandb
```bash
wandb login
```

### 3. Run Experiments
```bash
python mask_main.py --dataset MNIST --model_type MLP --lambda_reg 0.01
```

## 📊 Tracked Metrics

### Training Metrics
- **training/loss**: Loss value per epoch
- **training/accuracy**: Accuracy per epoch
- **training/epoch**: Current epoch number
- **training/label**: True label of the sample

### Sample Metrics
- **sample/sample_id**: Sequential sample number
- **sample/label**: True label of the sample
- **sample/correct**: Whether the sample was correctly predicted
- **sample/total_accuracy**: Cumulative accuracy across samples

### Mask Metrics (per mask)
- **mask_X/train_accuracy**: Training set accuracy for mask X
- **mask_X/test_accuracy**: Test set accuracy for mask X
- **mask_X/incorrect_accuracy**: Incorrect samples accuracy for mask X
- **mask_X/mask_mean**: Mean value of mask X parameters
- **mask_X/mask_std**: Standard deviation of mask X parameters

### Overall Metrics
- **overall/avg_train_accuracy**: Average training accuracy across all masks
- **overall/avg_test_accuracy**: Average test accuracy across all masks
- **overall/avg_incorrect_accuracy**: Average incorrect samples accuracy across all masks
- **overall/avg_mask_mean**: Average mask parameter mean across all masks
- **overall/avg_mask_std**: Average mask parameter std across all masks
- **overall/total_samples**: Total number of samples processed
- **overall/correct_predictions**: Total number of correctly predicted samples

## 🎯 Experiment Configuration

Each experiment tracks these hyperparameters:
- `dataset`: Dataset name (MNIST, CIFAR10, MED)
- `model_type`: Model architecture (MLP, AlexNet)
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate for mask optimization
- `lambda_reg`: Regularization strength
- `mask_initial_type`: Mask initialization type
- `num_masks`: Number of masks (10)
- `device`: Computing device (cuda/cpu)

## 📈 Dashboard Features

### 1. Real-time Monitoring
- Watch training progress in real-time
- Monitor loss curves and accuracy trends
- Track mask parameter evolution

### 2. Comparison Tools
- Compare different lambda_reg values
- Analyze mask initialization strategies
- Evaluate model architectures

### 3. Visualization
- Automatic plot generation for mask performance
- Interactive charts for hyperparameter sweeps
- Custom dashboard creation

## 🔧 Advanced Usage

### Custom Runs
```python
import wandb

# Custom experiment tracking
wandb.init(
    project="mask-regularization",
    name="custom_experiment",
    config={
        "custom_param": 0.5,
        "experiment_type": "ablation"
    }
)

# Log custom metrics
wandb.log({
    "custom/metric": value,
    "custom/step": step
})

wandb.finish()
```

### Hyperparameter Sweeps
Create a sweep configuration file:
```yaml
program: mask_main.py
method: grid
metric:
  name: overall/avg_incorrect_accuracy
  goal: maximize
parameters:
  lambda_reg:
    values: [0.001, 0.01, 0.1, 1.0]
  mask_lr:
    values: [0.001, 0.01, 0.1]
```

Run the sweep:
```bash
wandb sweep sweep_config.yaml
wandb agent <sweep_id>
```

## 📝 Tips for Better Tracking

1. **Meaningful Run Names**: Use descriptive names that include key parameters
2. **Consistent Tags**: Tag related experiments for easy grouping
3. **Save Important Artifacts**: Log model weights, plots, and results
4. **Document Experiments**: Add notes about experimental decisions

## 🐛 Troubleshooting

### Common Issues

1. **Import Error**: Make sure wandb is installed
   ```bash
   pip install wandb
   ```

2. **Login Issues**: Re-authenticate
   ```bash
   wandb login --relogin
   ```

3. **Project Not Found**: Create project in wandb dashboard or use existing project name

4. **Permission Errors**: Check wandb account permissions

### Getting Help
- Wandb Documentation: https://docs.wandb.ai/
- Community Forum: https://wandb.ai/forum
- GitHub Issues: https://github.com/wandb/wandb/issues

## 📊 Example Dashboard

Your wandb dashboard will show:
- Training curves for each mask
- Comparison of different regularization strengths
- Mask parameter distributions
- Accuracy trends across datasets
- Hyperparameter sensitivity analysis

Happy experimenting! 🎉
