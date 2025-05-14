import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1) Definisci il modello PyTorch
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
        x = F.relu(self.conv1(x))      # [batch,32,28,28]
        x = F.max_pool2d(x, 2)         # [batch,32,14,14]
        x = F.relu(self.conv2(x))      # [batch,64,14,14]
        x = F.max_pool2d(x, 2)         # [batch,64,7,7]
        # Flatten
        x = x.view(-1, 64 * 7 * 7)     # [batch,64*7*7]
        # Fully-connected
        x = F.relu(self.fc1(x))        # [batch,128]
        x = self.fc2(x)                # [batch,10]
        return x

# 2) Carica i dati (qui ogni client prende i primi 1000 esempi)
def load_train_data():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(".", train=True, download=True, transform=transform)
    subset = torch.utils.data.Subset(dataset, list(range(1000)))
    return torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)

# 3) Caricamento dati di test
def load_test_data():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(".", train=False, download=True, transform=transform)
    subset = torch.utils.data.Subset(dataset, list(range(200)))
    return torch.utils.data.DataLoader(subset, batch_size=32, shuffle=False)

# 3) Implementa il client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = Net()
        self.trainloader = load_train_data()
        self.testloader  = load_test_data()

    # Nota: ora ricevi anche `config`, che puoi ignorare se non ti serve
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # Imposta i pesi inviati dal server
        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(self.model.state_dict().keys(), parameters)
        }
        self.model.load_state_dict(state_dict, strict=True)

        # Allenamento locale
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        self.model.train()


        # Se si vogliono eseguire più epoche: for epoch in range(1, self.num_epochs + 1):
        # poi nella definizione del client client = FlowerClient(num_epochs=5)
        total_loss, correct, total = 0.0, 0, 0
        for data, target in self.trainloader:
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()


            total_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)

            # Calcola medie
            avg_loss = total_loss / total
            accuracy = correct / total

            # Log sul client
            print(f"[Client] round, loss={avg_loss:.4f}, accuracy={accuracy:.4f}")

        # Ritorna i nuovi parametri
        return self.get_parameters({}), total, {"loss": avg_loss, "accuracy": accuracy}

    def evaluate(self, parameters, config):
        # Carica i pesi
        state_dict = {
            k: torch.tensor(v)
            for k, v in zip(self.model.state_dict().keys(), parameters)
        }
        self.model.load_state_dict(state_dict, strict=True)

        # Valutazione locale
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for data, target in self.testloader:
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                preds = output.argmax(dim=1)
                correct += preds.eq(target).sum().item()
                total += target.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"[Client] evaluate → loss={avg_loss:.4f}, accuracy={accuracy:.4f}")

        return avg_loss, total, {"loss": avg_loss, "accuracy": accuracy}

if __name__ == "__main__":
    # Crea il NumPyClient e avvialo con la nuova API
    numpy_client = FlowerClient() # client = FlowerClient(num_epochs=5)
    fl.client.start_client(
        server_address="localhost:8080",
        client=numpy_client.to_client(),
    )
