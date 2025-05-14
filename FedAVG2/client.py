import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 1) Modello PyTorch identico al server
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

# 2) Caricamento dati MNIST (train + test small subset)
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(
        ".", train=True, download=True, transform=transform
    )
    testset = datasets.MNIST(
        ".", train=False, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(trainset, range(1000)),
        batch_size=32,
        shuffle=True,
    )
    testloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(testset, range(200)),
        batch_size=32,
        shuffle=False,
    )
    return trainloader, testloader

# 3) Client Flower con logging esteso
class LoggingClient(fl.client.NumPyClient):
    def __init__(self, cid: str):
        self.cid = cid
        self.model = Net()
        self.trainloader, self.testloader = load_data()

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters | config: {config}")
        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        print(f"[Client {self.cid}] Sending {len(params)} parameter arrays")
        return params

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit | Received parameters, config: {config}")
        # Carica pesi dal server
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

        # Training locale
        print(f"[Client {self.cid}] Starting local training")
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch_idx, (data, target) in enumerate(self.trainloader, 1):
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            preds = output.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += target.size(0)
            if batch_idx % 10 == 0:
                print(f"[Client {self.cid}] Batch {batch_idx}/{len(self.trainloader)} | "
                      f"loss={(total_loss/total):.4f}, acc={(correct/total):.4f}")

        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"[Client {self.cid}] fit complete | avg_loss={avg_loss:.4f}, accuracy={accuracy:.4f}")

        # Invio parametri aggiornati
        updated_params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        print(f"[Client {self.cid}] Uploading updated parameters")
        return updated_params, total, {"loss": avg_loss, "accuracy": accuracy}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate | Received parameters, config: {config}")
        # Carica pesi
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

        # Valutazione locale
        print(f"[Client {self.cid}] Starting evaluation")
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.testloader, 1):
                output = self.model(data)
                loss = criterion(output, target)
                total_loss += loss.item() * data.size(0)
                preds = output.argmax(dim=1)
                correct += preds.eq(target).sum().item()
                total += target.size(0)
                if batch_idx % 5 == 0:
                    print(f"[Client {self.cid}] Eval batch {batch_idx}/{len(self.testloader)} | "
                          f"loss={(total_loss/total):.4f}, acc={(correct/total):.4f}")

        avg_loss = total_loss / total
        accuracy = correct / total
        print(f"[Client {self.cid}] evaluate complete | avg_loss={avg_loss:.4f}, accuracy={accuracy:.4f}")

        return avg_loss, total, {"loss": avg_loss, "accuracy": accuracy}

if __name__ == "__main__":
    # Leggi un id client da CLI o default
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=str, default="0")
    args = parser.parse_args()
    
    client = LoggingClient(cid=args.cid)
    fl.client.start_client(
        server_address="localhost:8080",
        client=client.to_client(),
    )
