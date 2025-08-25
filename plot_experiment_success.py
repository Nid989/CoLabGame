import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Set professional plotting style
plt.style.use("default")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def get_experiments(plot_type):
    """Return experiment list based on plot type"""
    if plot_type == "modalities":
        return ["0_single_agent_a11y_tree_l1", "1_single_agent_screenshot_a11y_tree_l1", "2_single_agent_screenshot_l1"]
    elif plot_type == "single_vs_multi":
        return [
            "1_single_agent_screenshot_a11y_tree_l1",
            "5_multi_agent_star_screenshot_a11y_tree_l1",
            "8_multi_agent_blackboard_screenshot_a11y_tree_l1",
            "11_multi_agent_mesh_screenshot_a11y_tree_l1",
        ]
    else:
        raise ValueError("Invalid plot type. Use 'modalities' or 'single_vs_multi'")


def get_title(plot_type):
    """Return title based on plot type"""
    if plot_type == "modalities":
        return "Modalities Comparison"
    elif plot_type == "single_vs_multi":
        return "Single vs. Multi-Agent Comparison"
    else:
        return "Experiment Comparison"


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate experiment comparison plots")
    parser.add_argument("--type", choices=["modalities", "single_vs_multi"], default="modalities", help="Type of comparison to plot")
    args = parser.parse_args()

    # Read the CSV data
    df = pd.read_csv("results/raw.csv")

    # Get experiments and title based on plot type
    target_experiments = get_experiments(args.type)
    plot_title = get_title(args.type)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(plot_title, fontsize=14, fontweight="bold", y=0.98)

    # Professional color palette
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"][: len(target_experiments)]

    # Plot 1: Success Count
    success_df = df[(df["metric"] == "Success") & (df["experiment"].isin(target_experiments))]
    experiment_success = success_df.groupby("experiment")["value"].sum().reset_index()
    experiment_success = experiment_success.sort_values("experiment")

    bars1 = ax1.bar(
        range(len(experiment_success)), experiment_success["value"], color=colors, width=0.6, edgecolor="#1a1a1a", linewidth=0.8, alpha=0.85
    )

    ax1.set_xlabel("Experiments", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Success Count", fontsize=11, fontweight="bold")
    ax1.set_title("Success Rate", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(len(experiment_success)))
    ax1.set_xticklabels(experiment_success["experiment"], rotation=45, ha="right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax1.set_ylim(0, max(experiment_success["value"]) * 1.15)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, height + 0.05, f"{int(height)}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # Plot 2: Request Count
    request_df = df[(df["metric"] == "Request Count") & (df["experiment"].isin(target_experiments))]
    experiment_requests = request_df.groupby("experiment")["value"].mean().reset_index()
    experiment_requests = experiment_requests.sort_values("experiment")

    bars2 = ax2.bar(
        range(len(experiment_requests)), experiment_requests["value"], color=colors, width=0.6, edgecolor="#1a1a1a", linewidth=0.8, alpha=0.85
    )

    ax2.set_xlabel("Experiments", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Average Request Count", fontsize=11, fontweight="bold")
    ax2.set_title("Average Requests per Episode", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(len(experiment_requests)))
    ax2.set_xticklabels(experiment_requests["experiment"], rotation=45, ha="right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax2.set_ylim(0, max(experiment_requests["value"]) * 1.15)

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height + 0.2, f"{height:.1f}", ha="center", va="bottom", fontweight="bold", fontsize=9)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Show the plot
    plt.show()

    # Print the data for verification
    print(f"{plot_title}:")
    print("Success count by experiment:")
    print(experiment_success)
    print("\nAverage request count by experiment:")
    print(experiment_requests)


if __name__ == "__main__":
    main()
