import flwr as fl
from flwr.common import ndarrays_to_parameters, EvaluateRes
import torch
import torch.nn as nn
from flwr.server.strategy import FedAvg

# 1) Definizione dello stesso modello del client
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return self.fc(x.view(-1, 28 * 28))

# 2) Costruzione dei pesi iniziali (tutti a zero)
initial_model = Net()
initial_weights = [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
initial_parameters = ndarrays_to_parameters(initial_weights)

# 3) Funzione per aggregare metriche in modo pesato
def weighted_average(metrics):
    """Aggrega loss e accuracy pesando per numero di esempi."""
    # metrics: List[(num_examples, {"loss":..., "accuracy":...})]
    num_examples, losses, accs = zip(*[
        (num, m["loss"], m["accuracy"]) for num, m in metrics
    ])
    total = sum(num_examples)
    return {
        "loss": sum(l * n for l, n in zip(losses, num_examples)) / total,
        "accuracy": sum(a * n for a, n in zip(accs, num_examples)) / total,
    }

# 4) Strategia custom che stampa sempre fit ed evaluate
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


# 5) Configurazione della strategia con vincoli e funzioni di aggregazione
strategy = CustomFedAvg(
    initial_parameters=initial_parameters,
    min_available_clients=3,
    min_fit_clients=3,
    fraction_fit=1.0,
    min_evaluate_clients=3,
    fraction_evaluate=1.0,
    fit_metrics_aggregation_fn=weighted_average,
    evaluate_metrics_aggregation_fn=weighted_average,
)

# 6) Avvio del server
if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
