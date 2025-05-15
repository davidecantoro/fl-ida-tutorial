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

---
## 🎯 Tasks

Before you get started, complete the following exercises. Solutions are provided in the corresponding folders: `ShallowNN/`, `CNN/`, and `CustomFedAVG/`.

### Task 1 – Shallow NN with FedAvg

* **Objective:**
  Set up a Flower server and a client using the FedAvg strategy with a shallow neural network (single fully-connected layer).

* **Requirements:**
  The model should take a 28×28 MNIST image as input and output 10 class scores.

---

#### Client Steps

1. **Define the Shallow Model**

   * Single `Linear(28*28 → 10)` layer
   * `forward`: flatten the input to shape `(-1, 28*28)` then apply the linear layer

2. **Load and Subset MNIST Data**

   * Apply `transforms.ToTensor()`
   * Download and load the full MNIST training and test sets
   * Create a `trainloader` over a subset of **1 000** examples, `batch_size=32`, `shuffle=True`
   * Create a `testloader` over a subset of **200** examples, `batch_size=32`, `shuffle=False`

3. **Implement the Flower `NumPyClient`**

   * **`__init__`**: initialize `cid`, model, and data loaders
   * **`get_parameters`**: return model weights as NumPy arrays
   * **`fit`**:

     * Load global parameters into the local model
     * Train for **1 epoch** using SGD (`lr=0.01`) and `CrossEntropyLoss`
     * Return updated weights, number of training examples, and metrics (loss, accuracy)
   * **`evaluate`**:

     * Load global parameters
     * Evaluate on local test data, returning loss, number of test examples, and metrics

---

#### Server Steps

1. **Model Definition**

   * Define the same `Net` architecture as on the client (a single fully-connected layer)

2. **Configure the FedAvg Strategy**

   * Provide `initial_parameters` from the untrained model
   * Set `fraction_fit`, `min_fit_clients`, `min_available_clients`, `fraction_evaluate`, and `min_evaluate_clients` as needed

3. **Start the Flower Server**

   * Use `fl.server.start_server(...)` with your `FedAvg` strategy and a fixed number of rounds

---

### Task 2 – Lightweight CNN with FedAvg**

* **Objective:** Replace the shallow network with a small convolutional neural network and repeat the FedAvg experiment.
* **CNN:**
   1. **Input**
      * Single-channel MNIST images of size 1×28×28.

   2. **Conv-Block 1**

   * **Convolution:** 32 filters, 3×3 kernel, stride 1, padding 1 → preserves 28×28 spatial size.
   * **Activation:** ReLU introduces non-linearity.
   * **Pooling:** 2×2 max-pool → downsamples to 32 feature maps of size 14×14.

   3. **Conv-Block 2**

   * **Convolution:** 64 filters, 3×3 kernel, stride 1, padding 1 → preserves 14×14.
   * **Activation:** ReLU.
   * **Pooling:** 2×2 max-pool → downsamples to 64 feature maps of size 7×7.

   4. **Flatten**

   * Reshape the tensor from (64, 7, 7) → a vector of length 64 × 7 × 7 = 3 136.

   5. **Fully-Connected Layers**

   * **Hidden layer:** 128 units with ReLU.
   * **Output layer:** 10 units (one per MNIST class), producing the raw class scores (logits).


---

### Task 3 – Custom FedAvg Logging**

* **Objective:** Override the FedAvg strategy so that the server prints both loss and accuracy at the end of each evaluation round.

* **Steps:**

  1. Implement a `weighted_average(metrics)` function that:

     * Unpacks `(num_examples, {"loss": ..., "accuracy": ...})` tuples.
     * Calculates the total number of examples.
     * Returns a dict with weighted averages for both loss and accuracy.
  2. Override the `FedAvg` strategy by creating `CustomFedAvg`:

  * **override `aggregate_fit`:**

    1. Call the parent `aggregate_fit` to perform base aggregation.
    2. Extract aggregated **loss** and **accuracy** from the returned metrics.
    3. Print to console:

       ```
       [Server] Round {server_round} aggregate fit → loss={loss:.4f}, accuracy={accuracy:.4f}
       ```
    4. Return the original parameters and metrics.

  * **override `aggregate_evaluate`:**

    1. Call the parent `aggregate_evaluate` method.
    2. Extract `num_examples` and `metrics` (loss and accuracy) from the result.
    3. Print to console:

       ```
       [Server] Round {server_round} evaluate → accuracy={accuracy:.4f}, loss={loss:.4f}
       ```
    4. Return the original result to preserve Flower’s internal state.

3. Configure the server to use `CustomFedAvg` with both `fit_metrics_aggregation_fn` and `evaluate_metrics_aggregation_fn` set to `weighted_average`. Configure the server to use `CustomFedAvg` with both `fit_metrics_aggregation_fn` and `evaluate_metrics_aggregation_fn` set to `weighted_average`.


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
