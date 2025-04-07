import os
import matplotlib.pyplot as plt


def parse_moon_metrics(file_path):
    """Parses the MOON_global_metrics.log file to extract alpha values and final metrics."""
    metrics = {
        "alpha": [],
        "loss": [],
        "accuracy": [],
        "iou": [],
        "dice_coeff": [],
        "dice_loss": [],
        "moon_loss": [],
    }

    with open(file_path, "r") as f:
        lines = f.readlines()
        current_alpha = None

        for line in lines:
            line = line.strip()
            if line.startswith("Alpha:"):
                current_alpha = float(line.split(":")[1].strip())
            elif line.startswith("- Loss:") and current_alpha is not None:
                metrics["alpha"].append(current_alpha)
                metrics["loss"].append(float(line.split(":")[1].strip()))
            elif line.startswith("- Accuracy:") and current_alpha is not None:
                metrics["accuracy"].append(float(line.split(":")[1].strip()))
            elif line.startswith("- IoU:") and current_alpha is not None:
                metrics["iou"].append(float(line.split(":")[1].strip()))
            elif line.startswith("- Dice Coefficient:") and current_alpha is not None:
                metrics["dice_coeff"].append(float(line.split(":")[1].strip()))
            elif line.startswith("- Dice Loss:") and current_alpha is not None:
                metrics["dice_loss"].append(float(line.split(":")[1].strip()))
            elif line.startswith("- MOON Loss:") and current_alpha is not None:
                metrics["moon_loss"].append(float(line.split(":")[1].strip()))
                current_alpha = None  # Reset after capturing all metrics for this alpha

    return metrics


def plot_moon_metrics(metrics, output_folder):
    """Plots graphs for each metric with alpha on the x-axis for MOON metrics."""
    os.makedirs(output_folder, exist_ok=True)

    # Sort metrics by alpha
    sorted_indices = sorted(
        range(len(metrics["alpha"])), key=lambda i: metrics["alpha"][i]
    )
    sorted_metrics = {key: [metrics[key][i] for i in sorted_indices] for key in metrics}

    # Plot individual metrics (excluding moon_loss which gets its own special graph)
    for metric_name in [
        "loss",
        "accuracy",
        "iou",
        "dice_coeff",
        "dice_loss",
    ]:
        plt.figure(figsize=(10, 6))
        plt.plot(
            sorted_metrics["alpha"],
            sorted_metrics[metric_name],
            marker="o",
            color="blue",
            linewidth=2,
            label=metric_name.replace("_", " ").capitalize(),
        )
        plt.xlabel("Alpha (Non-IID Factor)", fontsize=12)
        plt.ylabel(metric_name.replace("_", " ").capitalize(), fontsize=12)
        plt.title(
            f"{metric_name.replace('_', ' ').capitalize()} vs Alpha in MOON",
            fontsize=14,
        )
        plt.grid(True)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{metric_name}.png"))
        plt.close()

    # Plot all metrics on a single graph (except for MOON Loss which has different scale)
    plt.figure(figsize=(12, 8))
    for metric_name in ["loss", "accuracy", "iou", "dice_coeff", "dice_loss"]:
        plt.plot(
            sorted_metrics["alpha"],
            sorted_metrics[metric_name],
            marker="o",
            linewidth=2,
            label=metric_name.replace("_", " ").capitalize(),
        )
    plt.xlabel("Alpha (Non-IID Factor)", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.title("All Metrics vs Alpha in MOON", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "all_metrics.png"))
    plt.close()

    # Plot the MOON Loss separately
    plt.figure(figsize=(10, 6))
    plt.plot(
        sorted_metrics["alpha"],
        sorted_metrics["moon_loss"],
        marker="o",
        color="purple",
        linewidth=2,
        label="MOON Loss",
    )
    plt.xlabel("Alpha (Non-IID Factor)", fontsize=12)
    plt.ylabel("MOON Loss", fontsize=12)
    plt.title("MOON Loss vs Alpha", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "moon_loss_only.png"))
    plt.close()

    # Add a section describing real-world differences
    plt.figure(figsize=(10, 6))
    plt.text(
        0.5,
        0.5,
        "Real-world vs. Simulation Differences:\n\n"
        "1. Data Redistribution: In this simulation, we redistribute samples\n"
        "   between partitions to ensure minimum samples. In real-world FL,\n"
        "   data cannot be moved between clients due to privacy constraints.\n\n"
        "2. Client Participation: All clients participate in every round here.\n"
        "   Real systems have dynamic client availability (mobile devices\n"
        "   might be offline or have low battery).\n\n"
        "3. Network Conditions: This simulation ignores bandwidth limitations,\n"
        "   latency, and communication failures common in real FL deployments.\n\n"
        "4. Heterogeneity: Real-world clients have varying compute capabilities,\n"
        "   while our simulation assumes homogeneous environments.\n\n"
        "5. Privacy Mechanisms: Production FL systems implement secure\n"
        "   aggregation and differential privacy, which we don't include here.",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=1", fc="white", ec="gray", alpha=0.9),
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "real_world_differences.png"))
    plt.close()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    moon_metrics_file = os.path.join(script_dir, "MOON_global_metrics.log")
    output_folder = os.path.join(script_dir, "moon_visualization")

    metrics = parse_moon_metrics(moon_metrics_file)
    plot_moon_metrics(metrics, output_folder)

    print("MOON visualization completed. Plots saved to:", output_folder)
