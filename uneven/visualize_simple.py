import os
import matplotlib.pyplot as plt


def parse_global_metrics(file_path):
    """Parses the global_metrics.log file to extract alpha values and final metrics."""
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


def plot_metrics(metrics, output_folder):
    """Plots graphs for each metric with alpha on the x-axis."""
    os.makedirs(output_folder, exist_ok=True)
    for metric_name in ["loss", "accuracy", "iou", "dice_coeff", "dice_loss"]:
        plt.figure()
        plt.plot(
            metrics["alpha"],
            metrics[metric_name],
            marker="o",
            label=metric_name.capitalize(),
        )
        plt.xlabel("Alpha")
        plt.ylabel(metric_name.capitalize())
        plt.title(f"{metric_name.capitalize()} vs Alpha")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_folder, f"{metric_name}.png"))
        plt.close()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    global_metrics_file = os.path.join(script_dir, "global_metrics.log")
    output_folder = os.path.join(script_dir, "simple_visualization")

    metrics = parse_global_metrics(global_metrics_file)
    plot_metrics(metrics, output_folder)
