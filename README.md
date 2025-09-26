# FedP2PAvg: A Peer-to-Peer Collaborative Framework for Federated Learning in Non-IID Scenarios

[![Paper @ ICANN 2025](https://img.shields.io/badge/ICANN%202025-Accepted-blue)]([https://link_to_paper_if_available](https://link.springer.com/chapter/10.1007/978-3-032-04558-4_31))

This repository contains the official **notebook-based implementation** of the paper:

**"FedP2PAvg: A Peer-to-Peer Collaborative Framework for Federated Learning in Non-IID Scenarios"**,  
accepted at the *International Conference on Artificial Neural Networks (ICANN 2025)*.

---

## 📌 Overview

**FedP2PAvg** is a federated learning algorithm that introduces a **peer-to-peer (P2P) refinement phase** in each training round.  
- **FedAvg baseline:** clients train locally on their own partitions → models are averaged by the central server.  
- **FedP2PAvg extension:** before aggregation, each client’s model is **trained for a short phase on another client’s data** (randomly or via a selection policy).  

This **extra collaboration step** reduces local bias and accelerates convergence, especially under **non-IID and imbalanced splits**.

---

## 🧠 Key Features

- **Peer-to-peer refinement** between clients before aggregation  
- **Non-IID partitioning** via Dirichlet distribution  
- **Support for multiple datasets:** MNIST, Fashion-MNIST, CIFAR-10  
- **Plug-and-play models:** CNNs tailored for each dataset (with GroupNorm or BatchNorm for stability)  
- **Colab-ready notebook:** easy to run and modify, no custom runners required  

---

## 🧪 Experimental Setup

- **Partitioning:** Dirichlet distribution, α = 0.1 (default)  
- **Clients:** 10 clients (configurable)  
- **Local training:**  
  - Optimizer: SGD  
  - Recommended for CIFAR-10: `lr=0.1`, `momentum=0.9`,
  - Recommended for MNIST / Fashion-MNIST: `lr=0.1`, `momentum=0.5`,
  - Local epochs: 2-10
- **Models:**  
  - **MNIST / Fashion-MNIST:** 2 conv + FC baseline (`NetMnist`)  
  - **CIFAR-10:** simplified CNN with GroupNorm and Global Average Pooling (`SimpleFLNetCifar10`)  
- **Evaluation:** centralized test set after each round  

---

## 🚀 Running the Notebook

Open [`FedP2PAvg_Hackathon.ipynb`](./FedP2PAvg_Hackathon.ipynb) in **Google Colab** or locally:

1. Select the dataset and run mode in the config cell:
   ```python
   RUN_MODE = "fedp2pavg"   # or "fedavg"
   DATASET  = "cifar10"     # "mnist", "fashionmnist", "cifar10"
2.Adjust hyperparameters (e.g., GLOBAL_ROUNDS, LOCAL_EPOCHS, DIRICHLET_ALPHA).
3. Run all cells.
4. Monitor training logs and plots to compare FedAvg vs FedP2PAvg.

If you use this code, please cite:
@inproceedings{fernandes2025fedp2pavg,
  title={FedP2PAvg: A Peer-to-Peer Collaborative Framework for Federated Learning in Non-IID Scenarios},
  author={Fernandes, Bruno J. T. and Freire, Agostinho and de Andrade, João V. R. and Silva, Leandro H. S. and Navarro-Guerrero, Nicolás},
  booktitle={34th International Conference on Artificial Neural Networks (ICANN)},
  year={2025}
}

📬 Contact
Bruno Fernandes: bruno.fernandes@upe.br

Supported by CAPES, FACEPE, CNPq, and the Alexander von Humboldt Foundation.

