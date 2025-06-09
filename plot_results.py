import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_losses_and_perplexities(logs_root: str = "logs", save_dir: str = None):
    logs_path = Path(logs_root)
    # find all metrics.csv under directories matching *_sweep*
    metrics_files = list(logs_path.glob("*FINAL*/*metrics.csv"))
    if not metrics_files:
        print(f"No metrics.csv found under {logs_root}")
        return

    dfs = []
    for mf in metrics_files:
        df = pd.read_csv(mf)
        # ensure epoch column
        if 'epoch' not in df.columns:
            df = df.rename(columns={df.columns[0]: 'epoch'})
        # derive model name
        model_name = mf.parent.name.split("_sweep")[0]
        df['model_name'] = model_name
        # compute perplexities
        df['train_ppl'] = np.exp(df['mean_train_loss'])
        df['dev_ppl'] = np.exp(df['mean_dev_loss'])
        dfs.append(df[['epoch', 'model_name', 'mean_train_loss',
                   'mean_dev_loss', 'train_ppl', 'dev_ppl']])
    all_df = pd.concat(dfs, ignore_index=True)

    # aggregate mean/std across runs for each model and epoch
    agg = all_df.groupby(['model_name', 'epoch']).agg(
        mean_train_loss_mean=('mean_train_loss', 'mean'),
        mean_train_loss_std=('mean_train_loss', 'std'),
        mean_dev_loss_mean=('mean_dev_loss', 'mean'),
        mean_dev_loss_std=('mean_dev_loss', 'std'),
        train_ppl_mean=('train_ppl', 'mean'),
        train_ppl_std=('train_ppl', 'std'),
        dev_ppl_mean=('dev_ppl', 'mean'),
        dev_ppl_std=('dev_ppl', 'std'),
    ).reset_index()

    # metrics to plot
    metrics = [
        ('mean_train_loss', 'mean_train_loss_mean',
         'mean_train_loss_std', 'Train Loss'),
        ('mean_dev_loss', 'mean_dev_loss_mean', 'mean_dev_loss_std', 'Dev Loss'),
        ('train_ppl', 'train_ppl_mean', 'train_ppl_std', 'Train Perplexity'),
        ('dev_ppl', 'dev_ppl_mean', 'dev_ppl_std', 'Dev Perplexity'),
    ]

    for metric_key, mean_col, std_col, title in metrics:
        plt.figure()
        for model_name, grp in agg.groupby('model_name'):
            x = grp['epoch']
            y = grp[mean_col]
            std = grp[std_col].fillna(0)
            plt.plot(x, y, label=model_name)
            plt.fill_between(x, y-std, y+std, alpha=0.2)
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend(fontsize='small')
        if not save_dir is None:
            plt.savefig(save_dir + metric_key + '_results.pdf')
        plt.show()


def summarize_perplexities(logs_root: str = "logs"):
    logs_path = Path(logs_root)
    metrics_files = list(logs_path.glob("*FINAL*/*metrics.csv"))

    if not metrics_files:
        raise FileNotFoundError(f"No metrics.csv under {logs_root}")

    # collect all runs
    dfs = []
    for mf in metrics_files:
        df = pd.read_csv(mf)
        if "epoch" not in df.columns:
            df = df.rename(columns={df.columns[0]: "epoch"})
        model = mf.parent.name.split("_sweep")[0]
        df = df.assign(
            model_name=model,
            train_ppl=np.exp(df["mean_train_loss"]),
            dev_ppl=np.exp(df["mean_dev_loss"])
        )
        dfs.append(df[["epoch", "model_name", "train_ppl", "dev_ppl"]])
    all_df = pd.concat(dfs, ignore_index=True)

    # pick the final epoch for each model
    out = []
    for model, grp in all_df.groupby("model_name"):
        last = grp.loc[grp["epoch"].idxmax()]
        run_vals = grp[grp["epoch"] == last["epoch"]]
        out.append({
            "model_name": model,
            "epoch":     int(last["epoch"]),
            "train_ppl_mean":     run_vals["train_ppl"].mean(),
            "train_ppl_median":   run_vals["train_ppl"].median(),
            "train_ppl_variance": run_vals["train_ppl"].var(),
            "dev_ppl_mean":       run_vals["dev_ppl"].mean(),
            "dev_ppl_median":     run_vals["dev_ppl"].median(),
            "dev_ppl_variance":   run_vals["dev_ppl"].var(),
        })

    return pd.DataFrame(out)


if __name__ == "__main__":
    summary_df = summarize_perplexities("logs")
    print(summary_df.to_markdown(index=False))
    # Call the plotting function
    plot_losses_and_perplexities("logs", "figs/")
