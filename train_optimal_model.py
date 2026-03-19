import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader as TorchDataLoader, random_split, Subset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from dataload import DataLoader, MultiTaskDataset
from model import CascadedModel
from util import r2_score_func, MetricTracker
from plot import plot_training_history

best_config = {
    'model_type': 'cascade',
    'fusion_type': 'cross_attention',
    'lr': 0.0005439591236568526,
    'dropout': 0.007625738023228556,
    'hidden_dims': {
        's1': [48, 48],
        's2': [32, 16],
        's3': [48, 48],
        's4': [16, 16]
    }
}

MAX_EPOCHS = 500


class ALMWeightController:
    def __init__(self, num_constraints=3, zeta=0.99, eta=0.005, eps=1e-8):
        self.lambdas = torch.zeros(num_constraints)
        self.v = torch.zeros(num_constraints)
        self.zeta = zeta
        self.eta = eta
        self.eps = eps
        self.mu = torch.ones(num_constraints) * 0.4

    def get_total_loss(self, objective_loss, constraint_losses):
        device = objective_loss.device
        self.lambdas = self.lambdas.to(device)
        self.mu = self.mu.to(device)

        c_theta = torch.stack(constraint_losses)
        linear_term = torch.sum(self.lambdas * c_theta)
        penalty_term = torch.sum((self.mu / 2.0) * (c_theta ** 2))

        return objective_loss + linear_term + penalty_term

    @torch.no_grad()
    def update_parameters(self, constraint_losses):
        c_theta = torch.stack(constraint_losses).detach().cpu()
        self.v = self.zeta * self.v + (1.0 - self.zeta) * (c_theta ** 2)
        self.mu = self.eta / (torch.sqrt(self.v) + self.eps)
        self.lambdas = self.lambdas + self.mu * c_theta


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

if __name__ == "__main__":
    print(f"\nStarting ALM adaptive loss training...")

    model = CascadedModel(
        stage_configs,
        hidden_dims=best_config['hidden_dims'],
        dropout_rate=best_config['dropout'],
        fusion_type=best_config['fusion_type']
    )

    print(f"Model built: CASCADE ({best_config['fusion_type']}) | Total params: {count_parameters(model)}")
    print(f"Hyperparams: LR={best_config['lr']:.6f}, Dropout={best_config['dropout']:.5f}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=best_config['lr'], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)

    alm_controller = ALMWeightController(num_constraints=3)

    history = []

    print("Starting ALM constrained training...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_tracker = MetricTracker()

        epoch_c1, epoch_c2, epoch_c3 = [], [], []

        for x_dict_batch, y_true_dict in train_loader:
            optimizer.zero_grad()

            preds_dict = model(x_dict_batch)

            loss_s1 = criterion(preds_dict['stage1'], y_true_dict['stage1'])
            loss_s2 = criterion(preds_dict['stage2'], y_true_dict['stage2'])
            loss_s3 = criterion(preds_dict['stage3'], y_true_dict['stage3'])
            loss_s4 = criterion(preds_dict['stage4'], y_true_dict['stage4'])

            total_loss = alm_controller.get_total_loss(loss_s4, [loss_s1, loss_s2, loss_s3])

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

            epoch_c1.append(loss_s1.item())
            epoch_c2.append(loss_s2.item())
            epoch_c3.append(loss_s3.item())

        mean_c1 = torch.tensor(np.mean(epoch_c1))
        mean_c2 = torch.tensor(np.mean(epoch_c2))
        mean_c3 = torch.tensor(np.mean(epoch_c3))
        alm_controller.update_parameters([mean_c1, mean_c2, mean_c3])

        model.eval()
        val_tracker = MetricTracker()
        with torch.no_grad():
            for x_dict_batch, y_true_dict in val_loader:
                preds_dict = model(x_dict_batch)

                v_loss_s1 = criterion(preds_dict['stage1'], y_true_dict['stage1'])
                v_loss_s2 = criterion(preds_dict['stage2'], y_true_dict['stage2'])
                v_loss_s3 = criterion(preds_dict['stage3'], y_true_dict['stage3'])
                v_loss_s4 = criterion(preds_dict['stage4'], y_true_dict['stage4'])

                v_total_loss = alm_controller.get_total_loss(v_loss_s4, [v_loss_s1, v_loss_s2, v_loss_s3])

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

        epoch_metrics = {
            "epoch": epoch, "lr": current_lr,
            "lambda_1": alm_controller.lambdas[0].item(),
            "lambda_2": alm_controller.lambdas[1].item(),
            "lambda_3": alm_controller.lambdas[2].item(),
            "mu_1": alm_controller.mu[0].item(),
            "mu_2": alm_controller.mu[1].item(),
            "mu_3": alm_controller.mu[2].item(),
        }
        for k, v in train_res.items(): epoch_metrics[f"train_{k}"] = v
        for k, v in val_res.items(): epoch_metrics[f"val_{k}"] = v
        history.append(epoch_metrics)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train R2: {train_res['total_r2']:.4f} | Val R2: {val_res['total_r2']:.4f}")
            print(f"          > ALM lambda: [{epoch_metrics['lambda_1']:.3f}, {epoch_metrics['lambda_2']:.3f}, {epoch_metrics['lambda_3']:.3f}]")

    csv_filename = "training_history_alm_cascade.csv"
    history_df = pd.DataFrame(history)
    history_df.to_csv(csv_filename, index=False)
    print(f"ALM training complete. History saved to {csv_filename}")

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
            preds_dict = model(x_dict_batch)

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

            out_csv = "last_50_predictions_alm_with_raw.csv"
            final_df.to_csv(out_csv, index=False)

            print(f"Last {num_to_predict} samples saved to: {out_csv}")
            break
