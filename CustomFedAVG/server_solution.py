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


# 3 BIS: weighted average
# - estrarre il numero di esempi, loss, accuracy -> usando zip
# - calcolo somma totale dei pesi
# - calcolo media pesata
def weighted_average(metrics):
    """Aggrega loss e accuracy pesando per numero di esempi."""
    # metrics: List[(num_examples, {"loss":..., "accuracy":...})]

    # in input ho
    # (n1, n2, …),   # num_examples
    # (l1, l2, …),   # losses
    # (a1, a2, …)    # accs
    # con zip lo trasformo in 
    # [(n1, l1, a1),
    #  (n2, l2, a2),
    #  …]
    num_examples, losses, accs = zip(*[
        (num, m["loss"], m["accuracy"]) for num, m in metrics
    ])
    total = sum(num_examples)
    return {
        "loss": sum(l * n for l, n in zip(losses, num_examples)) / total,
        "accuracy": sum(a * n for a, n in zip(accs, num_examples)) / total,
    }


# 4 BIS: Classe Custom FedAVG

# - override di aggregate_fit: qua si può modificare la strategia di aggregazione.
# – - chiamata al padre aggregate_fit per fare l'aggregazione
# – - recupero la loss
# – - recupero l'accuract
# – - print a schermo
# – - return parametri e metriche

# - override di aggregate_evaluate
# – - chiamata al padre
# – - estrazione metriche
# – - recupero la loss
# – - recupero l'accuract
# – - print a schermo
# - - return res
class CustomFedAvg(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # results: List of FitRes from clients
        parameters_agg, metrics_agg = super().aggregate_fit(server_round, results, failures)
        # metrics_agg non è più None grazie a fit_metrics_aggregation_fn
        loss = metrics_agg["loss"]
        acc  = metrics_agg["accuracy"]
        print(f"[Server] Round {server_round} aggregate fit → loss={loss:.4f}, accuracy={acc:.4f}")
        return parameters_agg, metrics_agg

    def aggregate_evaluate(self, server_round, results, failures):
        # Chiamata al parent; può restituire None, un EvaluateRes, o una tupla (num_examples, metrics)
        res = super().aggregate_evaluate(server_round, results, failures)
        if res is None:
            return None

        # Estrai metrics e num_examples sia se è un namedtuple sia se è una tupla
        metrics = getattr(res, "metrics", None)
        num_examples = getattr(res, "num_examples", None)
        if metrics is None:
            # Caso tupla: (num_examples, metrics)
            num_examples, metrics = res

        # Stampa accuracy e (se presente) loss
        acc = metrics.get("accuracy", 0.0)
        loss = metrics.get("loss", float("nan"))
        print(f"[Server] Round {server_round} evaluate → accuracy={acc:.4f}, loss={loss:.4f}")

        # Ritorna l’oggetto originale in modo che Flower mantenga il suo stato interno
        return res

# 3) Strategy FedAvg
# 5 BIS: CustomFedAVG: fit metrics, evaluate metrics
strategy = CustomFedAvg(
    initial_parameters=initial_parameters,
    min_available_clients=3,
    min_fit_clients=3,
    fraction_fit=1.0,
    min_evaluate_clients=3,
    fraction_evaluate=1.0,
    fit_metrics_aggregation_fn=weighted_average,        # <--
    evaluate_metrics_aggregation_fn=weighted_average,   # <--
)


# main
if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
