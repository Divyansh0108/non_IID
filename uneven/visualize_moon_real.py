import os
import matplotlib.pyplot as plt


def parse_moon_real_metrics(file_path):
    """Parses the MOON_real.log file to extract alpha values and final metrics."""
    metrics = {
        "alpha": [],
        "loss": [],
        "accuracy": [],
        "iou": [],
        "dice_coeff": [],
        "dice_loss": [],
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
                current_alpha = None  # Reset after capturing all metrics for this alpha

    return metrics


def plot_real_metrics(metrics, output_folder):
    """Plots graphs for each metric with alpha on the x-axis."""
    os.makedirs(output_folder, exist_ok=True)

    # Sort metrics by alpha
    sorted_indices = sorted(
        range(len(metrics["alpha"])), key=lambda i: metrics["alpha"][i]
    )
    sorted_metrics = {key: [metrics[key][i] for i in sorted_indices] for key in metrics}

    # Plot individual metrics
    for metric_name in ["loss", "accuracy", "iou", "dice_coeff", "dice_loss"]:
        plt.figure(figsize=(10, 6))
        plt.plot(
            sorted_metrics["alpha"],
            sorted_metrics[metric_name],
            marker="o",
            linewidth=2,
            color="blue",
            label=metric_name.replace("_", " ").capitalize(),
        )
        plt.xlabel("Alpha (Data Heterogeneity)", fontsize=12)
        plt.ylabel(metric_name.replace("_", " ").capitalize(), fontsize=12)
        plt.title(
            f"{metric_name.replace('_', ' ').capitalize()} vs Alpha in Real-World MOON",
            fontsize=14,
        )
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{metric_name}_real.png"))
        plt.close()

    # Plot combined view
    plt.figure(figsize=(12, 8))
    metrics_to_plot = {
        "accuracy": "Accuracy",
        "iou": "IoU",
        "dice_coeff": "Dice Coefficient",
    }

    for metric_name, label in metrics_to_plot.items():
        plt.plot(
            sorted_metrics["alpha"],
            sorted_metrics[metric_name],
            marker="o",
            linewidth=2,
            label=label,
        )

    plt.xlabel("Alpha (Data Heterogeneity)", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.title(
        "Performance Metrics vs Alpha in Real-World Federated Learning", fontsize=14
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "combined_real_metrics.png"))
    plt.close()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    moon_real_file = os.path.join(script_dir, "MOON_real.log")
    output_folder = os.path.join(script_dir, "moon_real_visualization")

    metrics = parse_moon_real_metrics(moon_real_file)
    plot_real_metrics(metrics, output_folder)

    print("Real-world MOON visualization completed. Plots saved to:", output_folder)
