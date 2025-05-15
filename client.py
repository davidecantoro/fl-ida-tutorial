
import flwr as fl  # Libreria Flower

# Torch: librerie per l'addestramento di reti neurali
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms # per la gestione dei dataset


# 1) definizione del modello locale
# - NN con input 28x28, outpu 10

# 2) caricamento dei dati
    # - caricamento di MNIST
    # -  DataLoader: gestisce il flusso degli esempi di un dataset verso il modello --> carica i dati

# 3) creazione della classe: Flower Client
    # componenti principali: 
    # - inizializzazione della classe 
    # - get parametri
    # - fase di fit (training locale) --> SGD, lr 0.01, CrossEntropyLoss
    # - - CNN: aggiungere momentum=0.9 ad ottimizzatore
    # - fase di evaluate (valutazione modello globale sui client)
class LoggingClient(fl.client.NumPyClient):
    def __init__(self, cid: str):


    def get_parameters(self, config):

        return params

    def fit(self, parameters, config):
        
        return updated_params, total, {"loss": avg_loss, "accuracy": accuracy}

    def evaluate(self, parameters, config):
        
        return avg_loss, total, {"loss": avg_loss, "accuracy": accuracy}


# main
if __name__ == "__main__":
    # Leggi un id client da CLI
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=str, default="0")
    args = parser.parse_args()
    