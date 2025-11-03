# FLASH-IMMUNE: A Zero-Trust Federated Learning Simulation

![Status](https://img.shields.io/badge/status-complete-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Frameworks](https://img.shields.io/badge/Frameworks-PyTorch%2C%20NetworkX-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This project is a Python-based simulation of **FLASH (Federated Learning for Smart Contract Hardening)** and its zero-trust extension, **FLASH-IMMUNE**.

This simulation models a decentralized network of nodes training a smart contract vulnerability detector. It demonstrates how the FLASH-IMMUNE framework successfully defends against malicious, Byzantine (poisoning) attackers by using an on-chain inspired reputation and validation system.

---

## 🏛️ Core Concepts

* **FLASH**: A privacy-preserving federated learning framework where nodes train on their private smart contract bytecode. Instead of sharing sensitive data, they only share model learnings.
* **The Problem**: The standard federated model trusts all nodes. Malicious nodes can send "poisoned" updates to corrupt the global model.
* **FLASH-IMMUNE**: A zero-trust security layer that *validates* every update before aggregation. It uses simulated cryptographic fingerprints and a novel reputation-scoring mechanism to automatically detect, reject, and quarantine malicious nodes, creating a self-healing network.

## ✨ Key Simulation Features

This code provides a high-fidelity simulation of the entire end-to-end system, including:

* **Federated Learning**: Simulates `N` clients training a PyTorch model on their local, private data.
* **Byzantine Attack Simulation**: A percentage of clients (e.g., 25%) are designated as "malicious" and actively try to poison the global model.
* **Zero-Trust Validation**: Implements the core FLASH-IMMUNE defense logic:
    * **Simulated ZKP**: A stub `simulate_zkp` function mimics the pass/fail check of a ZKP verifier.
    * **Statistical Fingerprinting**: Uses the **L2-Norm (magnitude)** of model deltas as a statistical fingerprint to identify poisoned updates as massive outliers.
    * **Reputation Scoring**: Implements the `R(n,t)` formula to dynamically update node reputations based on historical performance, consensus, and validity.
* **Automated Quarantine**: Nodes whose reputations fall below a threshold are automatically quarantined and barred from future rounds.
* **Cross-Chain Architecture**:
    * Simulates nodes split across multiple **Sidechains**.
    * Simulates an **IBC (Inter-Blockchain Communication)** protocol to sync sidechain models with the global model.
    * Simulates **Bitcoin Anchoring** by logging the global model's checksum.
    * Simulates **Lightning Micropayments** as a reward for valid contributions.
* **Self-Healing Rollback**: The system monitors for sudden accuracy drops (e.g., a successful attack) and automatically rolls back to the last known-good model.

---

## 🔬 Core Formulas Implemented

This simulation directly implements the key mathematical formulas from the FLASH and FLASH-IMMUNE research.

### FLASH: Local Training & Privacy

* **Federated Local Training Loss at Node k:**
    $$L_k(w) = \frac{1}{|D_k|} \sum_{(x_i,y_i) \in D_k} \ell(M_k(x_i; w), y_i)$$
* **Differential Privacy (Noise Addition):**
    $$\tilde{w}_k = w_k + \mathcal{N}(0, \sigma^2 I)$$

### FLASH-IMMUNE: Zero-Trust Validation

* **Node Reputation Scoring:**
    $$R(n,t) = \alpha \cdot H(n,t-1) + \beta \cdot C(n,t) + \gamma \cdot V(n,t)$$
    * `H`: Historical Contribution
    * `C`: Consensus Agreement (based on L2-Norm similarity)
    * `V`: Validity of Cryptographic Proofs (ZKP)

* **Byzantine-Resilient Global Aggregation:**
    $$w_{t+1} = \sum_{k \in \text{Trusted}} \frac{n_k \cdot R(k,t)}{\sum_{j \in \text{Trusted}} n_j \cdot R(j,t)} w_k^{(t)}$$

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* `pip` and `venv`
* Docker Desktop (Optional, for Method 2)

### Method 1: Run Locally (Recommended)

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/SamparkBhol/FLASH.git](https://github.com/SamparkBhol/FLASH.git)
    cd FLASH
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the simulation:**
    ```bash
    python main.py
    ```

### Method 2: Run with Docker

1.  **Navigate to the `docker` directory:**
    ```bash
    cd FLASH/docker
    ```

2.  **Build and run the container:**
    ```bash
    docker-compose up --build
    ```

---

## 📊 Interpreting the Results

After the simulation finishes, it will print a final summary to your console and generate three outputs.

### 1. Console Output

* **Live Logs**: Watch the logs for key events:
    * `[IMMUNE] REJECTED update from Client X`: The defense is working.
    * `[IMMUNE] Client X ... QUARANTINED`: The node has been removed.
    * `[IBC-SIM]`: A cross-chain sync has occurred.
    * `[BTC-SIM]`: A new model has been "anchored" to the mainchain.
    * `[!!! ROLLBACK !!!]`: The self-healing mechanism was triggered.
* **Final Tables**: The console prints a comparative analysis table and a final reputation scorecard, proving the system's efficacy.

### 2. `flash_immune_simulation_results.png`

This file contains 6 plots. The two most important are:

* **Client Reputation Over Time (The "Money Shot")**: This graph is the primary proof. It will show a clean separation:
    * **Blue Lines (Benign Clients)**: Rise to and stay at a high reputation (`~1.0`).
    * **Red Lines (Malicious Clients)**: Instantly dive to a near-zero reputation (`~0.2`) and flatline.
* **Update Rejection per Round**: This graph shows *why* the reputations separated. It will show a massive **red spike** in the first few rounds as the system detects, rejects, and quarantines all 6 attackers. The **green line (Benign Rejected)** will stay flat at 0, proving the system does not have an "auto-immune" problem.

### 3. `flash_network_topology.png`

This image shows the final state of the network. It's color-coded by reputation:
* **Green**: Healthy, active nodes.
* **Red**: Quarantined malicious nodes.

This visually confirms that the self-healing network has successfully identified and isolated all threats.

### 4. Turtle Splash Screen

A final summary window will open, presenting the final accuracy and attack rejection rate, confirming the simulation's success.

## outputs

<img width="1087" height="872" alt="image" src="https://github.com/user-attachments/assets/17aaa494-b4bd-4dd0-8dca-db683c334a06" />
<img width="875" height="853" alt="image" src="https://github.com/user-attachments/assets/ac6ba417-73a7-4d47-ac0c-7f3fa3ffb4c6" />
```
set_seed(42)
```

output is attached 
