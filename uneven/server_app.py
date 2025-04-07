"""fedapp: A Flower / PyTorch app."""

import os
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from uneven.task import UNet, get_weights

script_dir = os.path.dirname(os.path.abspath(__file__))


def server_fn(context: Context):
    """Defines the server-side logic for federated learning."""
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    ndarrays = get_weights(UNet())
    parameters = ndarrays_to_parameters(ndarrays)

    class LoggingFedAvg(FedAvg):
        """Custom FedAvg strategy with logging capabilities."""

        def aggregate_fit(self, rnd, results, failures):
            aggregated_result = super().aggregate_fit(rnd, results, failures)
            if aggregated_result:
                parameters, metrics = aggregated_result  # Unpack the tuple
                if metrics:
                    print(
                        f"Round {rnd} - Training Loss: {metrics.get('loss', 'N/A')}, "
                        f"Training Accuracy: {metrics.get('accuracy', 'N/A')}"
                    )
                return aggregated_result

        def aggregate_evaluate(self, rnd, results, failures):
            aggregated_result = super().aggregate_evaluate(rnd, results, failures)
            if aggregated_result:
                parameters, metrics = aggregated_result  # Unpack the tuple
                if metrics:
                    testing_loss = metrics.get("loss", "N/A")
                    testing_accuracy = metrics.get("accuracy", "N/A")
                    testing_iou = metrics.get("iou", "N/A")
                    testing_dice_coeff = metrics.get("dice_coeff", "N/A")
                    testing_dice_loss = metrics.get("dice_loss", "N/A")
                    print(
                        f"Round {rnd} - Testing Loss: {testing_loss}, "
                        f"Testing Accuracy: {testing_accuracy}, "
                        f"Testing IoU: {testing_iou}, "
                        f"Testing DiceCoeff: {testing_dice_coeff}, "
                        f"Testing DiceLoss: {testing_dice_loss}"
                    )
                return aggregated_result

    strategy = LoggingFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
