import matplotlib.pyplot as plt
import pandas as pd

def plot_training_history(history_df: pd.DataFrame):
    tasks = ['stage1', 'stage2', 'stage3', 'stage4']

    plt.figure(figsize=(10, 6))
    plt.plot(history_df["epoch"], history_df["train_total_loss"], label="Train Total Loss", color="blue")
    plt.plot(history_df["epoch"], history_df["val_total_loss"], label="Val Total Loss", color="orange")
    plt.title("Overall Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("history_total_loss.png")
    plt.close()

    if "train_total_r2" in history_df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(history_df["epoch"], history_df["train_total_r2"], label="Train Mean R2", color="green")
        plt.plot(history_df["epoch"], history_df["val_total_r2"], label="Val Mean R2", color="red")

        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label="Perfect Score (1.0)")
        plt.axhline(y=0.0, color='black', linestyle='-', alpha=0.2)

        plt.title("Overall Model R2 Score (Mean of all stages)")
        plt.xlabel("Epoch")
        plt.ylabel("R2 Score")

        min_y = max(history_df["val_total_r2"].min() - 0.1, -1.0)
        plt.ylim([min_y, 1.1])

        plt.legend()
        plt.grid(True)
        plt.savefig("history_total_r2.png")
        plt.close()

    for task in tasks:
        if f"train_{task}_loss" not in history_df.columns:
            continue

        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.plot(history_df["epoch"], history_df[f"train_{task}_loss"], label="Train Loss", color="blue")
        plt.plot(history_df["epoch"], history_df[f"val_{task}_loss"], label="Val Loss", color="orange")
        plt.title(f"{task.upper()} - Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(history_df["epoch"], history_df[f"train_{task}_r2"], label="Train R2", color="green")
        plt.plot(history_df["epoch"], history_df[f"val_{task}_r2"], label="Val R2", color="red")

        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        plt.axhline(y=0.0, color='black', linestyle='-', alpha=0.2)

        plt.title(f"{task.upper()} - R2 Score")
        plt.xlabel("Epoch")
        plt.ylabel("R2")

        min_task_y = max(history_df[f"val_{task}_r2"].min() - 0.1, -1.0)
        plt.ylim([min_task_y, 1.1])

        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"history_{task}.png")
        plt.close()
