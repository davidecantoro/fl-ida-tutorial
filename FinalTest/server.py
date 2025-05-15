import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
import torch
import torch.nn as nn

# 1) Definizione modello identico al client
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)
    def forward(self, x):
        return self.fc(x.view(-1, 28 * 28))

# 2) Pesi iniziali (tutti a zero)
initial_model = Net()
initial_weights = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
initial_parameters = ndarrays_to_parameters(initial_weights)

# 3) Strategy FedAvg
# - % client che parteciperanno alla federazione
# - numero minimo di client che dovranno partecipare alla federazione
# - numero minimo di client che dovranno essere connessi
# - % di client che parteciperanno alla federazione
# - numero min di client che parteciperanno all'evaluate 
strategy = FedAvg(
    initial_parameters=initial_parameters,
    fraction_fit=1.0,
    min_fit_clients=3,
    min_available_clients=3,
    fraction_evaluate=0.5,
    min_evaluate_clients=2,
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
