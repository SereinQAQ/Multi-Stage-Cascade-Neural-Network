import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader as TorchDataLoader, random_split, Subset
import pandas as pd
import numpy as np

from dataload import DataLoader, MultiTaskDataset
from model import CascadedModel, PureMLPModel, OriginalCascadedModel
from util import r2_score_func, MetricTracker
from plot import plot_training_history

MODEL_TYPE = 'cascade'

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

if MODEL_TYPE == 'cascade':
    my_hidden_dims = {'s1': [32], 's2': [32], 's3': [32], 's4': [32]}
    model = CascadedModel(stage_configs, hidden_dims=my_hidden_dims, fusion_type='attention')
elif MODEL_TYPE == 'pure_mlp':
    model = PureMLPModel(stage_configs)
elif MODEL_TYPE == 'original_cascade':
    model = OriginalCascadedModel(stage_configs)
else:
    raise ValueError("MODEL_TYPE must be 'cascade', 'pure_mlp', or 'original_cascade'")

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
epochs = 500
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

history = []
tasks = ['stage1', 'stage2', 'stage3', 'stage4']

for epoch in range(1, epochs + 1):
    model.train()
    train_tracker = MetricTracker()

    for x_dict_batch, y_true_dict in train_loader:
        optimizer.zero_grad()

        if MODEL_TYPE in ['cascade', 'original_cascade']:
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
            if MODEL_TYPE in ['cascade', 'original_cascade']:
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
                    'stage1': preds_5dim[:, 0:1],
                    'stage2': preds_5dim[:, 1:3],
                    'stage3': preds_5dim[:, 3:4],
                    'stage4': preds_5dim[:, 4:5]
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

csv_filename = f"training_history_{MODEL_TYPE}.csv"
history_df = pd.DataFrame(history)
history_df.to_csv(csv_filename, index=False)
print(f"[{MODEL_TYPE.upper()}] Training complete. History saved to {csv_filename}")

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
        if MODEL_TYPE in ['cascade', 'original_cascade']:
            preds_dict = model(x_dict_batch)
        else:
            preds_5dim = model(x_dict_batch)
            preds_dict = {
                'stage1': preds_5dim[:, 0:1],
                'stage2': preds_5dim[:, 1:3],
                'stage3': preds_5dim[:, 3:4],
                'stage4': preds_5dim[:, 4:5]
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
