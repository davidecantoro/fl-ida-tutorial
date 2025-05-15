# FL-IDA Tutorial

A step-by-step tutorial to get started with Federated Learning (FL) using Flower and PyTorch.

## 📖 Overview

This repository demonstrates a complete Federated Learning workflow:

* **Flower Server** implementing the Federated Averaging strategy.
* **PyTorch Clients** training locally on the MNIST dataset.

You will learn how to:

1. Set up the development environment.
2. Install required dependencies.
3. Launch the server and multiple clients.
4. Run a federated learning experiment and monitor global metrics (loss and accuracy).

## 📋 Prerequisites

* Python 3.8 or higher
* [Conda](https://docs.conda.io/) (optional but recommended)

## 🛠️ Installation

1. **Clone the repository (or manual download it)**

   ```bash
   git clone https://github.com/davidecantoro/fl-ida-tutorial.git
   cd fl-ida-tutorial
   ```
2. **Create and activate the Conda environment**

   ```bash
   conda env create -f environment.yaml
   conda activate fl-learning
   ```

## 🔧 Repository Structure

```
fl-ida-tutorial/
├── environment.yaml    # Conda environment definition
├── server.py           # Flower server
├── client.py           # Flower client
└── README.md           # This file
```
## 🎯 Tasks

Before you get started, complete the following exercises. Solutions are provided in the corresponding folders: `ShallowNN/`, `CNN/`, and `CustomFedAVG/`.

**Task 1 – Shallow NN with FedAvg**

* **Objective:** Set up a Flower server and one or more clients using the FedAvg strategy with a shallow neural network (single fully-connected layer).
* **Requirements:** The model should accept a 28×28 MNIST image and output 10 class scores.
* **Steps:**

  1. Open the `ShallowNN/` folder and review the provided server and client scripts.
  2. Verify that `fl.server.start_server(...)` is used on the server side and `fl.client.start_client(...)` on the client side.
  3. Run a federated experiment and confirm that global loss decreases round by round.

**Task 2 – Lightweight CNN with FedAvg**

* **Objective:** Replace the shallow network with a small convolutional neural network (two Conv2d+ReLU+MaxPool blocks, followed by two fully-connected layers) and repeat the FedAvg experiment.
* **Steps:**

  1. Open the `CNN/` folder and locate the `Net` class defining the CNN.
  2. Ensure server and client scripts import this CNN model instead of the shallow one.
  3. Launch the experiment for multiple rounds and compare the global loss and accuracy against Task 1.

**Task 3 – Custom FedAvg Logging**

* **Objective:** Extend or override the FedAvg strategy so that the server prints both loss and accuracy at the end of each evaluation round.
* **Steps:**

  1. Open the `CustomFedAVG/` folder.
  2. In the strategy configuration, add an `evaluate_metrics_aggregation_fn` that computes the weighted average of client accuracies.
  3. Capture the `History` object returned by `start_server()` and print `history.metrics_distributed["accuracy"]` alongside loss each round.

---

## 🚀 Getting Started

### 1) Start the Flower Server

```bash
python server.py
```

The server will listen on `localhost:8080` and coordinate training rounds.

### 2) Launch Federated Clients

Open multiple terminal windows or tabs. For each client, run:

```bash
python client.py --cid 0
python client.py --cid 1
python client.py --cid 2
```

Replace the `--cid` value to assign a unique client ID (e.g., 0, 1, 2).

### 3) Monitor the Experiment

* The server console will display aggregated metrics (loss and accuracy) each round.
* Each client prints detailed logs of local training and evaluation.

## 📈 Experiment Walkthrough

1. Configure the environment and dependencies.
2. Start the server in one terminal.
3. Launch 2–3 clients in separate terminals with distinct `--cid` values.
4. Observe global loss and accuracy evolving over rounds.
5. Modify local training settings (e.g., number of epochs, learning rate) in `client.py` and restart to see the impact.
