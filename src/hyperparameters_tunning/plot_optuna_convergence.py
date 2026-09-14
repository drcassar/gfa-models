import os
import sys
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)


NAMES = [
    r"sqlite:///GFA_binary_FEATENG/model/glass_formation_top_features_CV_RF_F1_macro_f1_score_macro_2000_TPESampler.db",
    r"sqlite:///GFA_binary_CHEM/model/glass_formation_cv_FE_RF_f1_score_macro_2000_TPESampler.db",
]

STUDY_NAMES = [
    "glass_formation_top_features_CV_RF_F1_macro",
    "glass_formation_cv_FE_RF",
]


def load_study(storage: str, study_name: str) -> optuna.Study:
    return optuna.load_study(study_name=study_name, storage=storage)


def get_trials_dataframe(study: optuna.Study) -> pd.DataFrame:

    trials = study.trials
    direction = study.direction  # MINIMIZE or MAXIMIZE

    records = []
    for t in trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            records.append({"trial_number": t.number, "value": t.value})

    df = pd.DataFrame(records).sort_values("trial_number").reset_index(drop=True)

    if direction == optuna.study.StudyDirection.MINIMIZE:
        df["best_so_far"] = df["value"].cummin()
    else:
        df["best_so_far"] = df["value"].cummax()

    return df, direction


def plot_history(studies_data: dict, save_path: str = "optuna_history.png"):
    """
    One subplot per study showing every trial value as a scatter point and
    the running best as a bold step line.
    """
    n = len(studies_data)
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(8, 4 * n),
        sharex=False,
        squeeze=False,
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (name, (df, direction)) in enumerate(studies_data.items()):
        ax = axes[idx, 0]
        color = colors[idx % len(colors)]

        # All trial values
        ax.scatter(
            df["trial_number"],
            df["value"],
            s=8,
            alpha=0.35,
            color=color,
            label="Trial value",
            zorder=2,
        )

        # Running best
        ax.step(
            df["trial_number"],
            df["best_so_far"],
            where="post",
            linewidth=2,
            color=color,
            label="Running best",
            zorder=3,
        )

        # Annotate final best
        best_val = df["best_so_far"].iloc[-1]
        best_trial = df.loc[df["best_so_far"] == best_val, "trial_number"].iloc[0]
        ax.axhline(best_val, color=color, linestyle="--", linewidth=0.8, alpha=0.6)
        ax.annotate(
            f"Best = {best_val:.4f}\n(trial {best_trial})",
            xy=(best_trial, best_val),
            xytext=(0.65, 0.85),
            textcoords="axes fraction",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1),
        )

        dir_label = (
            "Minimize"
            if direction == optuna.study.StudyDirection.MINIMIZE
            else "Maximize"
        )
        ax.set_title(f"{name}  ({dir_label})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Trial number", fontsize=10)
        ax.set_ylabel("Objective value", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
        ax.grid(True, linestyle="--", alpha=0.4)

    # fig.suptitle(
    #     "Hyperparameter Optimization History",
    #     fontsize=14,
    #     fontweight="bold",
    #     y=1.01,
    # )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")


def plot_convergence(
    studies_data: dict,
    save_path: str = "optuna_convergence.png",
):
    """
    Single-axis figure with one running-best curve per study.
    Suitable as a supplementary figure in the paper.

    The x-axis is normalized to [0, 1] (fraction of total trials) so that
    studies with different numbers of trials are comparable on the same scale.
    """
    fig, ax = plt.subplots(figsize=(5, 4.5))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (name, (df, direction)) in enumerate(studies_data.items()):
        color = colors[idx % len(colors)]

        x = df["trial_number"].values
        y = df["best_so_far"].values

        # Normalize x to fraction of total trials for comparability
        x_norm = x / x.max()

        if idx == 0:
            label = "CHEM"
        else:
            label = "FEATENG"

        ax.plot(
            x_norm,
            y,
            drawstyle="steps-post",
            linewidth=2,
            color=color,
            label=label,
        )

        # Mark the point of convergence (where improvement < 0.1% of range)
        improvement = np.abs(np.diff(y))
        value_range = y.max() - y.min() if y.max() != y.min() else 1.0
        converged_idx = np.where(improvement < 0.001 * value_range)[0]
        if len(converged_idx) > 0:
            conv_x = x_norm[converged_idx[0]]
            conv_y = y[converged_idx[0]]
            # ax.axvline(conv_x, color=color, linestyle=":", linewidth=1, alpha=0.7)
            # ax.annotate(
            #     f"~{conv_x*100:.0f}%",
            #     xy=(conv_x, conv_y),
            #     xytext=(conv_x + 0.03, conv_y),
            #     fontsize=8,
            #     color=color,
            # )

    ax.set(xlim=[0, 1])
    ax.set_xlabel("Fraction of trials completed", fontsize=11)
    ax.set_ylabel("Best objective value", fontsize=11)
    # ax.set_title(
    #     "Hyperparameter Optimization Convergence", fontsize=13, fontweight="bold"
    # )
    ax.legend(fontsize=10, framealpha=0.9, loc=4)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")


def plot_trial_diagnostics(
    study: optuna.Study, study_name: str, save_path: str = "optuna_diagnostics.png"
):
    """
    Bar chart of trial states. Useful to report in the supplementary material
    to show the optimizer ran cleanly without excessive pruning or failures.
    """
    from collections import Counter

    states = Counter(t.state.name for t in study.trials)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(
        states.keys(), states.values(), color=["#4C72B0", "#DD8452", "#C44E52"]
    )
    ax.bar_label(bars, fontsize=10)
    ax.set_title(f"Trial States — {study_name}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count", fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved: {save_path}")


def main():

    # Load data for each selected study
    studies_data = {}
    for storage, name in zip(NAMES, STUDY_NAMES):
        print(storage, name)
        study = load_study(storage, name)
        df, direction = get_trials_dataframe(study)
        n_complete = len(df)
        n_total = len(study.trials)
        print(
            f"\n{name}: {n_complete}/{n_total} complete trials | "
            f"Direction: {study.direction.name} | "
            f"Best: {study.best_value:.6f}"
        )
        studies_data[name] = (df, direction)

        # Per-study diagnostics
        plot_trial_diagnostics(
            study,
            name,
            save_path=f"optuna_diagnostics_{name}.png",
        )

    # Combined plots
    plot_history(studies_data, save_path="optuna_history.png")
    plot_convergence(studies_data, save_path="optuna_convergence.png")


if __name__ == "__main__":
    main()
