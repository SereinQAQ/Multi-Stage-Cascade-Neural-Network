import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader as TorchDataLoader, random_split
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt

from dataload import DataLoader, MultiTaskDataset
from model import CascadedModel, PureMLPModel
from util import r2_score_func

OPTIMIZATION_ALGO = 'NSGA2'
MODEL_TO_OPTIMIZE = 'cascade'
FIXED_FUSION_TYPE = 'attention'
N_TRIALS = 40
MAX_EPOCHS = 500


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("Loading data...")
loader = DataLoader()
X_dict, Y_targets = loader.get_features()

stage_configs = {stage: {"cat_dims": feats["cat_dims"], "num_dim": feats["num_dim"]}
                 for stage, feats in X_dict.items()}

full_dataset = MultiTaskDataset(X_dict, Y_targets)

train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = TorchDataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = TorchDataLoader(val_ds, batch_size=64)

tasks = ['stage1', 'stage2', 'stage3', 'stage4']


def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout', 0.0, 0.3)

    if MODEL_TO_OPTIMIZE == 'cascade':
        if FIXED_FUSION_TYPE == 'search':
            fusion_type = trial.suggest_categorical('fusion_type', ['concat', 'glu', 'attention', 'cross_attention'])
        else:
            fusion_type = FIXED_FUSION_TYPE

        my_hidden_dims = {}
        for stage in ['s1', 's2', 's3', 's4']:
            n_layers = trial.suggest_int(f'n_layers_{stage}', 1, 2)
            layer_dims = []
            for i in range(n_layers):
                dim = trial.suggest_int(f'n_units_{stage}_L{i}', 16, 64, step=16)
                layer_dims.append(dim)
            my_hidden_dims[stage] = layer_dims

        model = CascadedModel(
            stage_configs=stage_configs,
            hidden_dims=my_hidden_dims,
            dropout_rate=dropout_rate,
            fusion_type=fusion_type
        )

    elif MODEL_TO_OPTIMIZE == 'pure_mlp':
        n_layers_mlp = trial.suggest_int('n_layers_mlp', 1, 4)
        mlp_hidden_dims = []
        for i in range(n_layers_mlp):
            dim = trial.suggest_int(f'n_units_mlp_L{i}', 16, 64, step=16)
            mlp_hidden_dims.append(dim)

        model = PureMLPModel(
            stage_configs=stage_configs,
            hidden_dims=mlp_hidden_dims,
            dropout_rate=dropout_rate
        )

    else:
        raise ValueError("MODEL_TO_OPTIMIZE must be 'cascade' or 'pure_mlp'")

    param_count = count_parameters(model)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x_dict_batch, y_true_dict in train_loader:
            optimizer.zero_grad()

            if MODEL_TO_OPTIMIZE == 'cascade':
                preds_dict = model(x_dict_batch)
                losses = {t: criterion(preds_dict[t], y_true_dict[t]) for t in tasks}
                total_loss = sum(losses.values())
            else:
                preds_5dim = model(x_dict_batch)
                y_true_all = torch.cat([
                    y_true_dict['stage1'], y_true_dict['stage2'],
                    y_true_dict['stage3'], y_true_dict['stage4']
                ], dim=1)
                total_loss = criterion(preds_5dim, y_true_all)
                preds_dict = {
                    'stage1': preds_5dim[:, 0:1],
                    'stage2': preds_5dim[:, 1:3],
                    'stage3': preds_5dim[:, 3:4],
                    'stage4': preds_5dim[:, 4:5]
                }

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        if OPTIMIZATION_ALGO == 'BOHB' and epoch % 10 == 0:
            model.eval()
            val_r2_list = []
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    if MODEL_TO_OPTIMIZE == 'cascade':
                        p_val = model(x_val)
                    else:
                        p_5dim = model(x_val)
                        p_val = {
                            'stage1': p_5dim[:, 0:1],
                            'stage2': p_5dim[:, 1:3],
                            'stage3': p_5dim[:, 3:4],
                            'stage4': p_5dim[:, 4:5]
                        }
                    for t in tasks:
                        val_r2_list.append(r2_score_func(y_val[t], p_val[t]).item())

            mean_val_r2 = np.mean(val_r2_list)
            trial.report(mean_val_r2, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    model.eval()
    val_r2_list = []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            if MODEL_TO_OPTIMIZE == 'cascade':
                p_val = model(x_val)
            else:
                p_5dim = model(x_val)
                p_val = {
                    'stage1': p_5dim[:, 0:1],
                    'stage2': p_5dim[:, 1:3],
                    'stage3': p_5dim[:, 3:4],
                    'stage4': p_5dim[:, 4:5]
                }
            for t in tasks:
                val_r2_list.append(r2_score_func(y_val[t], p_val[t]).item())

    mean_val_r2 = np.mean(val_r2_list)

    if OPTIMIZATION_ALGO == 'NSGA2':
        return mean_val_r2, param_count
    else:
        return mean_val_r2


if __name__ == "__main__":
    print(f"Starting [{OPTIMIZATION_ALGO}] hyperparameter search...")
    print(f"Target model: {MODEL_TO_OPTIMIZE.upper()}")
    if MODEL_TO_OPTIMIZE == 'cascade':
        print(f"Fusion type: {FIXED_FUSION_TYPE}")

    if OPTIMIZATION_ALGO == 'TPE':
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    elif OPTIMIZATION_ALGO == 'BOHB':
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner(min_resource=10, max_resource=MAX_EPOCHS, reduction_factor=3)
        )
    elif OPTIMIZATION_ALGO == 'NSGA2':
        study = optuna.create_study(
            directions=["maximize", "minimize"],
            sampler=optuna.samplers.NSGAIISampler(seed=42, population_size=20)
        )
    else:
        raise ValueError("Invalid OPTIMIZATION_ALGO setting")

    study.optimize(objective, n_trials=N_TRIALS)

    print(f"\nSearch complete. {len(study.trials)} trials evaluated.")

    if OPTIMIZATION_ALGO in ['TPE', 'BOHB']:
        print(f"Best R2: {study.best_value:.4f}")
        print("Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"   - {key}: {value}")

        if OPTIMIZATION_ALGO == 'BOHB':
            pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            print(f"Hyperband pruned trials: {len(pruned_trials)}")

    elif OPTIMIZATION_ALGO == 'NSGA2':
        best_trials = study.best_trials
        print(f"Pareto front contains {len(best_trials)} elite configurations:")
        for i, trial in enumerate(best_trials):
            fusion_str = trial.params.get('fusion_type', FIXED_FUSION_TYPE) if MODEL_TO_OPTIMIZE == 'cascade' else 'pure_mlp'
            print(f"  [{i+1}] R2: {trial.values[0]:.4f} | Params: {trial.values[1]} | Fusion: {fusion_str}")

        all_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        all_r2 = [t.values[0] for t in all_trials]
        all_params = [t.values[1] for t in all_trials]
        pareto_r2 = [t.values[0] for t in best_trials]
        pareto_params = [t.values[1] for t in best_trials]

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        ax.scatter(all_params, all_r2, c='lightgray', s=40, alpha=0.6, label='Explored Architectures')
        ax.scatter(pareto_params, pareto_r2, c='#D62728', s=100, alpha=0.9, edgecolors='black', linewidth=1.5, label='Pareto Front (NSGA-II)')

        sorted_indices = np.argsort(pareto_params)
        ax.plot(np.array(pareto_params)[sorted_indices], np.array(pareto_r2)[sorted_indices], color='#D62728', linestyle='--', linewidth=1.5, alpha=0.5)

        ax.set_title(f"NAS: Accuracy vs. Complexity ({MODEL_TO_OPTIMIZE.upper()})", fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel("Model Complexity (Number of Parameters)", fontsize=13, fontweight='bold')
        ax.set_ylabel("Validation $R^2$ Score", fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', frameon=True, shadow=True, edgecolor='black').get_frame().set_facecolor('white')

        plt.tight_layout()
        filename = f"pareto_front_{MODEL_TO_OPTIMIZE}.png"
        plt.savefig(filename, bbox_inches='tight')
        print(f"\nPareto front plot saved to: {filename}")
