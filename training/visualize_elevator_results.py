import json
import os
import pandas as pd
import matplotlib.pyplot as plt

# ================================================================
# Utility Functions
# ================================================================

def load_results(results_file="evaluation_results.json"):
    """Load nested evaluation results JSON."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"{results_file} not found.")
    with open(results_file, "r") as f:
        return json.load(f)


def flatten_results(data):
    """Convert nested results dict into a flat pandas DataFrame."""
    rows = []
    for model, obs_dict in data.items():
        for obs_type, rew_dict in obs_dict.items():
            for reward_type, act_dict in rew_dict.items():
                for action_type, metrics in act_dict.items():
                    row = {
                        "model": model,
                        "observation": obs_type,
                        "reward": reward_type,
                        "action": action_type,
                        **metrics
                    }
                    rows.append(row)
    df = pd.DataFrame(rows)
    return df


# ================================================================
# Visualization Functions
# ================================================================

def plot_models_comparison(df, obs_type, reward_type, action_type, save_dir="plots"):
    """Compare all models under the same configuration."""
    os.makedirs(save_dir, exist_ok=True)
    subset = df[(df["observation"] == obs_type) &
                (df["reward"] == reward_type) &
                (df["action"] == action_type)]
    if subset.empty:
        print(f"No data found for {obs_type}-{reward_type}-{action_type}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["avg_passengers_completed", "avg_wait_time", "avg_journey_time"]
    titles = ["Passengers Completed ↑", "Average Wait Time ↓", "Average Journey Time ↓"]

    for ax, metric, title in zip(axes, metrics, titles):
        ax.bar(subset["model"], subset[metric])
        ax.set_title(f"{title}\n({obs_type}, {reward_type}, {action_type})", fontsize=10)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/models_comparison_{obs_type}_{reward_type}_{action_type}.png")
    plt.show()


def plot_model_combinations(df, model_name, save_dir="plots"):
    """Show how one model performs across all combinations."""
    os.makedirs(save_dir, exist_ok=True)
    model_df = df[df["model"] == model_name]
    if model_df.empty:
        print(f"No data found for {model_name}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ["avg_passengers_completed", "avg_wait_time", "avg_journey_time"]
    titles = ["Passengers Completed ↑", "Average Wait Time ↓", "Average Journey Time ↓"]

    x_labels = [f"{o[:1]}-{r[:1]}-{a[:1]}" for o, r, a in zip(model_df["observation"], model_df["reward"], model_df["action"])]

    for ax, metric, title in zip(axes, metrics, titles):
        ax.bar(x_labels, model_df[metric])
        ax.set_title(f"{title}\n{model_name}", fontsize=10)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_combinations.png")
    plt.show()


def plot_metric_heatmap(df, metric, save_dir="plots"):
    """Visualize performance as a heatmap for each model and combination."""
    import seaborn as sns
    os.makedirs(save_dir, exist_ok=True)

    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        if model_df.empty:
            continue

        # Create a pivot table
        pivot = model_df.pivot_table(
            index=["observation", "reward"],
            columns="action",
            values=metric
        )
        plt.figure(figsize=(6, 4))
        sns.heatmap(pivot, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"{model} - {metric.replace('_', ' ').title()}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{model}_heatmap_{metric}.png")
        plt.show()


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    results = load_results("evaluation_results.json")
    df = flatten_results(results)
    print("\nLoaded results into DataFrame:")
    print(df.head())

    # --- 1. Compare models under the same condition ---
    for obs in df["observation"].unique():
        for reward in df["reward"].unique():
            for action in df["action"].unique():
                plot_models_comparison(df, obs, reward, action)

    # --- 2. Compare all combinations for each model ---
    for model in df["model"].unique():
        plot_model_combinations(df, model)

    # --- 3. Heatmap comparisons per metric ---
    for metric in ["avg_passengers_completed", "avg_wait_time", "avg_journey_time"]:
        plot_metric_heatmap(df, metric)

    print("\nAll plots saved in ./plots/")
