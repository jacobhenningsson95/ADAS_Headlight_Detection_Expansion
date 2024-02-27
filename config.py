__config__ = {

    # dataset path
    'dataset_path': 'data\PVDN',
    'log_path': 'log.txt',

    'max_num_car': 8,
    'max_num_light': 8,

    # for model
    'nstack': 3,  # Number of stacks in the model (If GPU memory is insufficient, decrease this to 2)
    'inp_dim': 256,  # Input dimension
    'oup_dim': 64,  # Output dimension
    'increase': 128,  # Increase factor
    'input_res': 512,  # Input resolution
    'output_res': 128,  # Output resolution
    # for training
    'bn': True,  # Use batch normalization
    'negative_samples': True,  # Use samples where there is no vehicles
    'autocast': True,  # Use automatic mixed precision (AMP)
    'save_all_models': True,  # Save all models during training
    'weighted_dataset': True,  # Use weighted dataset during training
    'day_samples': True,  # Use extra day samples containing multiple vehicles in training
    'override_saved_config': True,  # Override configuration of saved model
    'batch_size': 12,  # Training batch size (validation batch size is 2 * batch_size)
    'epochs': 200,  # Number of training epochs
    'scheduler': 'CosineAnnealingLR',  # Learning rate scheduler type
    'min_lr': 1e-6,  # Minimum learning rate for scheduler
    'val_epoch': 1,  # Validate the model every "config['val_epoch']" epoch
    'learning_rate': 1e-5,  # Initial learning rate
    'num_workers': 4,  # Number of workers for data loading
    'model': 'Hourglass',
}