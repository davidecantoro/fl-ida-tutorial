import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters
import torch
import torch.nn as nn

# 1) Definizione modello identico al client
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Primo blocco convoluzionale
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # Secondo blocco convoluzionale
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Fully-connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Convoluzione + ReLU + MaxPool
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        # Fully-connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2) Pesi iniziali (tutti a zero)
initial_model = Net()
initial_weights = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
initial_parameters = ndarrays_to_parameters(initial_weights)

# 3) Strategy FedAvg
strategy = FedAvg(
    initial_parameters=initial_parameters,
    fraction_fit=1.0,
    min_fit_clients=3,
    min_available_clients=3,
    fraction_evaluate=0.5,
    min_evaluate_clients=2,
)

# main
if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
