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

# - - - - - - - - - - - - - 

# 3 BIS: weighted average
# - estrarre il numero di esempi, loss, accuracy -> usando zip
# - calcolo somma totale dei pesi
# - calcolo media pesata 

    # in input ho
    # (n1, n2, …),   # num_examples
    # (l1, l2, …),   # losses
    # (a1, a2, …)    # accs
    # con zip lo trasformo in 
    # [(n1, l1, a1),
    #  (n2, l2, a2),
    #  …]


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

# 5 BIS: CustomFedAVG: fit metrics, evaluate metrics
 

# main
if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )
