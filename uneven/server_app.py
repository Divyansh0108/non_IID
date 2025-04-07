"""fedapp: A Flower / PyTorch app."""

import os
import random
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

    class DynamicClientFedAvg(FedAvg):
        """Custom FedAvg strategy with dynamic client participation."""

        def __init__(self, fraction_fit, min_available_clients, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fraction_fit = fraction_fit
            self.min_available_clients = min_available_clients

        def configure_fit(self, server_round, parameters, client_manager):
            """Select a random subset of clients for training."""
            # Get all client IDs directly
            client_ids = list(client_manager.all())

            if not client_ids:
                return []

            num_clients = max(
                int(self.fraction_fit * len(client_ids)),
                self.min_available_clients,
            )
            # Make sure we don't try to sample more clients than available
            num_clients = min(num_clients, len(client_ids))

            # Sample client IDs randomly
            selected_client_ids = random.sample(client_ids, num_clients)

            # Create configuration for each selected client
            client_instructions = []
            for client_id in selected_client_ids:
                client_instructions.append((client_id, parameters))

            return client_instructions

    strategy = DynamicClientFedAvg(
        fraction_fit=fraction_fit,
        min_available_clients=2,
        initial_parameters=parameters,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
