import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader as TorchDataLoader, random_split, Subset
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt

from dataload import DataLoader, MultiTaskDataset
from model import CascadedModel, PureMLPModel, OriginalCascadedModel
from util import r2_score_func, MetricTracker
from plot import plot_training_history

RUN_MODE = 'train_best'

OPTIMIZATION_ALGO = 'NSGA2'
MODEL_TO_OPTIMIZE = 'cascade'
FIXED_FUSION_TYPE = 'attention'
N_TRIALS = 40
MAX_EPOCHS = 500

best_config = {
    'model_type': 'cascade',
    'fusion_type': 'concat',
    'lr': 0.0025470526233391183,
    'dropout': 0.05545633665765811,
    'hidden_dims': {
        's1': [64],
        's2': [32, 64],
        's3': [16],
        's4': [48]
    }
}


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
    if MODEL_TO_OPTIMIZE == 'all':
        current_model_type = trial.suggest_categorical('model_type', ['cascade', 'pure_mlp', 'original_cascade'])
    else:
        current_model_type = MODEL_TO_OPTIMIZE

    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout', 0.0, 0.3)

    if current_model_type in ['cascade', 'original_cascade']:
        my_hidden_dims = {}
        for stage in ['s1', 's2', 's3', 's4']:
            n_layers = trial.suggest_int(f'n_layers_{stage}', 1, 2)
            layer_dims = [trial.suggest_int(f'n_units_{stage}_L{i}', 16, 64, step=16) for i in range(n_layers)]
            my_hidden_dims[stage] = layer_dims

        if current_model_type == 'cascade':
            fusion_type = trial.suggest_categorical('fusion_type', ['concat', 'glu', 'attention', 'cross_attention']) if FIXED_FUSION_TYPE == 'search' else FIXED_FUSION_TYPE
            model = CascadedModel(stage_configs, hidden_dims=my_hidden_dims, dropout_rate=dropout_rate, fusion_type=fusion_type)
        else:
            model = OriginalCascadedModel(stage_configs, hidden_dims=my_hidden_dims, dropout_rate=dropout_rate)

    elif current_model_type == 'pure_mlp':
        n_layers_mlp = trial.suggest_int('n_layers_mlp', 1, 4)
        mlp_hidden_dims = [trial.suggest_int(f'n_units_mlp_L{i}', 16, 64, step=16) for i in range(n_layers_mlp)]
        model = PureMLPModel(stage_configs, hidden_dims=mlp_hidden_dims, dropout_rate=dropout_rate)

    param_count = count_parameters(model)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x_dict_batch, y_true_dict in train_loader:
            optimizer.zero_grad()
            if current_model_type in ['cascade', 'original_cascade']:
                preds_dict = model(x_dict_batch)
                losses = {t: criterion(preds_dict[t], y_true_dict[t]) for t in tasks}
                total_loss = sum(losses.values())
            else:
                preds_5dim = model(x_dict_batch)
                y_true_all = torch.cat([y_true_dict[t] for t in tasks], dim=1)
                total_loss = criterion(preds_5dim, y_true_all)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        if OPTIMIZATION_ALGO == 'BOHB' and epoch % 10 == 0:
            model.eval()
            val_r2_list = []
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    if current_model_type in ['cascade', 'original_cascade']:
                        p_val = model(x_val)
                    else:
                        p_5dim = model(x_val)
                        p_val = {'stage1': p_5dim[:, 0:1], 'stage2': p_5dim[:, 1:3], 'stage3': p_5dim[:, 3:4], 'stage4': p_5dim[:, 4:5]}
                    val_r2_list.extend([r2_score_func(y_val[t], p_val[t]).item() for t in tasks])
            trial.report(np.mean(val_r2_list), epoch)
            if trial.should_prune(): raise optuna.exceptions.TrialPruned()

    model.eval()
    val_r2_list = []
    with torch.no_grad():
        for x_val, y_val in val_loader:
            if current_model_type in ['cascade', 'original_cascade']:
                p_val = model(x_val)
            else:
                p_5dim = model(x_val)
                p_val = {'stage1': p_5dim[:, 0:1], 'stage2': p_5dim[:, 1:3], 'stage3': p_5dim[:, 3:4], 'stage4': p_5dim[:, 4:5]}
            val_r2_list.extend([r2_score_func(y_val[t], p_val[t]).item() for t in tasks])

    mean_val_r2 = np.mean(val_r2_list)
    return (mean_val_r2, param_count) if OPTIMIZATION_ALGO == 'NSGA2' else mean_val_r2


def train_best_model(config):
    print(f"\nStarting best model training...")
    model_type = config['model_type']

    if model_type == 'cascade':
        model = CascadedModel(stage_configs, hidden_dims=config['hidden_dims'],
                              dropout_rate=config['dropout'], fusion_type=config['fusion_type'])
    elif model_type == 'original_cascade':
        model = OriginalCascadedModel(stage_configs, hidden_dims=config['hidden_dims'], dropout_rate=config['dropout'])
    elif model_type == 'pure_mlp':
        model = PureMLPModel(stage_configs, hidden_dims=config['hidden_dims'], dropout_rate=config['dropout'])
    else:
        raise ValueError("Unsupported model type")

    print(f"Model built: {model_type.upper()} | Total params: {count_parameters(model)}")
    print(f"Hyperparams: LR={config['lr']:.5f}, Dropout={config['dropout']:.5f}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)

    history = []

    print("Starting training...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_tracker = MetricTracker()

        for x_dict_batch, y_true_dict in train_loader:
            optimizer.zero_grad()

            if model_type in ['cascade', 'original_cascade']:
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
                    'stage1': preds_5dim[:, 0:1], 'stage2': preds_5dim[:, 1:3],
                    'stage3': preds_5dim[:, 3:4], 'stage4': preds_5dim[:, 4:5]
                }

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_tracker.update("total_loss", total_loss.item())
            current_batch_r2_list = []
            for t in tasks:
                task_r2 = r2_score_func(y_true_dict[t], preds_dict[t]).item()
                task_loss = criterion(preds_dict[t], y_true_dict[t]).item()
                train_tracker.update(f"{t}_loss", task_loss)
                train_tracker.update(f"{t}_r2", task_r2)
                current_batch_r2_list.append(task_r2)
            train_tracker.update("total_r2", np.mean(current_batch_r2_list))

        model.eval()
        val_tracker = MetricTracker()
        with torch.no_grad():
            for x_dict_batch, y_true_dict in val_loader:
                if model_type in ['cascade', 'original_cascade']:
                    preds_dict = model(x_dict_batch)
                    v_losses = {t: criterion(preds_dict[t], y_true_dict[t]) for t in tasks}
                    v_total_loss = sum(v_losses.values())
                else:
                    preds_5dim = model(x_dict_batch)
                    y_true_all = torch.cat([
                        y_true_dict['stage1'], y_true_dict['stage2'],
                        y_true_dict['stage3'], y_true_dict['stage4']
                    ], dim=1)
                    v_total_loss = criterion(preds_5dim, y_true_all)
                    preds_dict = {
                        'stage1': preds_5dim[:, 0:1], 'stage2': preds_5dim[:, 1:3],
                        'stage3': preds_5dim[:, 3:4], 'stage4': preds_5dim[:, 4:5]
                    }

                val_tracker.update("total_loss", v_total_loss.item())
                val_batch_r2_list = []
                for t in tasks:
                    v_task_r2 = r2_score_func(y_true_dict[t], preds_dict[t]).item()
                    v_task_loss = criterion(preds_dict[t], y_true_dict[t]).item()
                    val_tracker.update(f"{t}_loss", v_task_loss)
                    val_tracker.update(f"{t}_r2", v_task_r2)
                    val_batch_r2_list.append(v_task_r2)
                val_tracker.update("total_r2", np.mean(val_batch_r2_list))

        train_res = train_tracker.result()
        val_res = val_tracker.result()

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        epoch_metrics = {"epoch": epoch, "lr": current_lr}
        for k, v in train_res.items(): epoch_metrics[f"train_{k}"] = v
        for k, v in val_res.items(): epoch_metrics[f"val_{k}"] = v
        history.append(epoch_metrics)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train R2: {train_res['total_r2']:.4f} | Val R2: {val_res['total_r2']:.4f} | LR: {current_lr:.6f}")

    csv_filename = f"training_history_{model_type}.csv"
    history_df = pd.DataFrame(history)
    history_df.to_csv(csv_filename, index=False)
    print(f"[{model_type.upper()}] Training complete. History saved to {csv_filename}")

    try:
        plot_training_history(history_df)
        print("Training curve plots generated successfully.")
    except Exception as e:
        print(f"Plot generation error: {e}")

    print("\nRunning inference on last 50 samples...")
    total_len = len(full_dataset)
    num_to_predict = min(50, total_len)
    last_50_indices = list(range(total_len - num_to_predict, total_len))

    last_50_dataset = Subset(full_dataset, last_50_indices)
    infer_loader = TorchDataLoader(last_50_dataset, batch_size=num_to_predict, shuffle=False)

    model.eval()
    with torch.no_grad():
        for x_dict_batch, _ in infer_loader:
            if model_type in ['cascade', 'original_cascade']:
                preds_dict = model(x_dict_batch)
            else:
                preds_5dim = model(x_dict_batch)
                preds_dict = {
                    'stage1': preds_5dim[:, 0:1], 'stage2': preds_5dim[:, 1:3],
                    'stage3': preds_5dim[:, 3:4], 'stage4': preds_5dim[:, 4:5]
                }

            column_names = {
                'stage1': ['Precursor Particle Size (nm) [Pred]'],
                'stage2': ['Cathode Particle Size (nm) [Pred]', 'I003/I104 [Pred]'],
                'stage3': ['HV [Pred]'],
                'stage4': ['Test Temp [Pred]']
            }

            results_df = pd.DataFrame()

            for t in tasks:
                scaler = loader.scalers_y[t]
                pred_np = preds_dict[t].numpy()
                inv_pred = scaler.inverse_transform(pred_np)
                for idx, col_name in enumerate(column_names[t]):
                    results_df[col_name] = inv_pred[:, idx]

            if loader.remove_duplicates:
                feature_cols = loader.categorical_cols + loader.numerical_cols
                feature_strings = loader.list_df.iloc[:, feature_cols].astype(str).agg('_'.join, axis=1)
                unique_indices = feature_strings.drop_duplicates().index
                df_raw_filtered = loader.list_df.loc[unique_indices].copy()
            else:
                df_raw_filtered = loader.list_df.copy()

            original_last_50 = df_raw_filtered.iloc[-num_to_predict:].reset_index(drop=True)
            final_df = pd.concat([original_last_50, results_df], axis=1)

            out_csv = "last_50_predictions_with_raw.csv"
            final_df.to_csv(out_csv, index=False)

            print(f"Last {num_to_predict} samples saved to: {out_csv}")
            print("\nPrediction preview (first 5 rows):")
            display_cols = list(original_last_50.columns[:3]) + list(results_df.columns)
            print(final_df[display_cols].head())
            break


if __name__ == "__main__":
    if RUN_MODE == 'train_best':
        train_best_model(best_config)

    elif RUN_MODE == 'search':
        print(f"Starting [{OPTIMIZATION_ALGO}] global search...")
        if OPTIMIZATION_ALGO == 'TPE':
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
        elif OPTIMIZATION_ALGO == 'BOHB':
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.HyperbandPruner(min_resource=10, max_resource=MAX_EPOCHS, reduction_factor=3))
        elif OPTIMIZATION_ALGO == 'NSGA2':
            study = optuna.create_study(directions=["maximize", "minimize"], sampler=optuna.samplers.NSGAIISampler(population_size=20))

        study.optimize(objective, n_trials=N_TRIALS)
        print(f"\nSearch complete. {len(study.trials)} trials evaluated.")
    else:
        print("Invalid RUN_MODE setting.")
