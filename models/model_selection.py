def select_model(model_val=1):
    model_args = {}
    if model_val == 1:
        model_args = {'model_name': 'model_1_A', 'model_dir': 'saved_models',
                      'model_subDir': 'feature_compressed',
                      'input_dim': 97, 'output_dim': 1, 'optimizer': 'adadelta',
                      'metrics': ["mean_absolute_error"], 'loss': "mse", 'earlyStop': True,
                      'weights': 'weights.h5', 'plot_loss': True,
                      'neurons': {'alpha': 1, 'beta': (1/2), 'gamma': (1/3)},
                      'n_layers': 4, 'weights_init': 'glorot_uniform', 'dropout': .25,
                      'epochs': 100, 'batch_size': 256, 'early_mon': 'val_mean_absolute_error',
                      'mode': 'min', 'checkout_mon': 'val_loss'}

    elif model_val == 2:
        model_args = {'model_name': 'model_1_B', 'model_dir': 'saved_models',
                      'model_subDir': 'feature_compressed',
                      'input_dim': 97, 'output_dim': 1, 'optimizer': 'adam',
                      'metrics': ["mean_absolute_error"], 'loss': "mse", 'earlyStop': True,
                      'weights': 'weights.h5', 'plot_loss': False,
                      'neurons': {'alpha': 1, 'beta': (1/2), 'gamma': (1/3)},
                      'n_layers': 4, 'weights_init': 'glorot_uniform', 'dropout': .25,
                      'epochs': 100, 'batch_size': 256, 'early_mon': 'val_mean_absolute_error',
                      'mode': 'min', 'checkout_mon': 'val_loss'}
    elif model_val == 3:
        model_args = {'model_name': 'model_1_C', 'model_dir': 'saved_models',
                      'model_subDir': 'feature_compressed',
                      'input_dim': 97, 'output_dim': 1, 'optimizer': 'adadelta',
                      'metrics': ["mean_absolute_error"], 'loss': "mse", 'earlyStop': True,
                      'weights': 'weights.h5', 'plot_loss': False,
                      'neurons': {'alpha': 1, 'beta': (1/2), 'gamma': (1/3)},
                      'n_layers': 4, 'weights_init': 'he_normal', 'dropout': .25,
                      'epochs': 100, 'batch_size': 256, 'early_mon': 'val_mean_absolute_error',
                      'mode': 'min', 'checkout_mon': 'val_loss'}
    elif model_val == 4:
        model_args = {'model_name': 'model_1_D', 'model_dir': 'saved_models',
                      'model_subDir': 'feature_compressed',
                      'input_dim': 97, 'output_dim': 1, 'optimizer': 'adam',
                      'metrics': ["mean_absolute_error"], 'loss': "mse", 'earlyStop': True,
                      'weights': 'weights.h5', 'plot_loss': False,
                      'neurons': {'alpha': 1, 'beta': (1/2), 'gamma': (1/3)},
                      'n_layers': 4, 'weights_init': 'he_normal', 'dropout': .25,
                      'epochs': 100, 'batch_size': 256, 'early_mon': 'val_mean_absolute_error',
                      'mode': 'min', 'checkout_mon': 'val_loss'}
    # ------------------------------------------------------------------------------------------------------------------
    elif model_val == 5:
        model_args = {'model_name': 'model_1_A', 'model_dir': 'saved_models',
                      'model_subDir': 'class_feature_compressed',
                      'input_dim': 97, 'output_dim': 3, 'optimizer': 'adadelta',
                      'metrics': ["accuracy"], 'loss': "categorical_crossentropy", 'earlyStop': True,
                      'weights': 'weights.h5', 'plot_loss': True,
                      'neurons': {'alpha': 1, 'beta': (1/2), 'gamma': (1/3)},
                      'n_layers': 4, 'weights_init': 'glorot_uniform', 'dropout': .25,
                      'epochs': 100, 'batch_size': 256, 'early_mon': 'val_acc',
                      'mode': 'max', 'checkout_mon': 'val_acc'}

    elif model_val == 6:
        model_args = {'model_name': 'model_2_B', 'model_dir': 'saved_models',
                      'model_subDir': 'class_feature_compressed',
                      'input_dim': 97, 'output_dim': 3, 'optimizer': 'adam',
                      'metrics': ["accuracy"], 'loss': "categorical_crossentropy", 'earlyStop': True,
                      'weights': 'weights.h5', 'plot_loss': False,
                      'neurons': {'alpha': 1, 'beta': (1/2), 'gamma': (1/3)},
                      'n_layers': 4, 'weights_init': 'glorot_uniform', 'dropout': .25,
                      'epochs': 100, 'batch_size': 256, 'early_mon': 'val_acc',
                      'mode': 'max', 'checkout_mon': 'val_acc'}

    elif model_val == 7:
        model_args = {'model_name': 'model_3_C', 'model_dir': 'saved_models',
                      'model_subDir': 'class_feature_compressed',
                      'input_dim': 97, 'output_dim': 3, 'optimizer': 'adadelta',
                      'metrics': ["accuracy"], 'loss': "categorical_crossentropy", 'earlyStop': True,
                      'weights': 'weights.h5', 'plot_loss': False,
                      'neurons': {'alpha': 1, 'beta': (1 / 2), 'gamma': (1 / 3)},
                      'n_layers': 4, 'weights_init': 'he_normal', 'dropout': .25,
                      'epochs': 100, 'batch_size': 256, 'early_mon': 'val_acc',
                      'mode': 'max', 'checkout_mon': 'val_acc'}
    return model_args


