# FedP2PAvg: A Peer-to-Peer Collaborative Framework for Federated Learning in Non-IID Scenarios

[![Paper @ ICANN 2025](https://img.shields.io/badge/ICANN%202025-Accepted-blue)](https://link_to_paper_if_available)

This repository contains the official implementation of the paper:

**"FedP2PAvg: A Peer-to-Peer Collaborative Framework for Federated Learning in Non-IID Scenarios"**,  
accepted at the *International Conference on Artificial Neural Networks (ICANN 2025)*.

## 📌 Overview

**FedP2PAvg** is a federated learning framework that introduces a peer-to-peer (P2P) refinement phase in each training round. Unlike classical approaches like FedAvg, which rely solely on local training followed by central aggregation, FedP2PAvg adds an extra layer of collaboration: each client's model is sent to a randomly selected peer for further training before the global update. This strategy reduces model bias and accelerates convergence in highly imbalanced and non-IID data settings.

## 🧠 Key Features

- **Peer-to-peer refinement** of local models before aggregation
- **Improved convergence** and accuracy in non-IID scenarios
- **Compatibility** with classical FedAvg architecture
- **Evaluation** on MNIST, Fashion-MNIST, and CIFAR-10 with Dirichlet α=0.1

## 📊 Results Summary

| Dataset        | Accuracy (FedP2PAvg) | Rounds to Target Accuracy |
|----------------|----------------------|----------------------------|
| MNIST          | 98.17%               | 16                         |
| Fashion-MNIST  | 84.35%               | 20                         |
| CIFAR-10       | 67.49%               | 125                        |

## 🧪 Experimental Setup

- **Data partitioning**: Dirichlet distribution with α = 0.1
- **Local training**: SGD optimizer, learning rate 0.01, momentum 0.5
- **Architectures**:
  - MNIST / Fashion-MNIST: 2 conv layers + 2 FC layers + dropout
  - CIFAR-10: VGG11
- **Rounds**:
  - MNIST & Fashion-MNIST: 140 rounds
  - CIFAR-10: 300 rounds

## 🛠️ Installation

Clone this repository:

```bash
git clone https://github.com/bjtf/FedP2PAvg.git
cd FedP2PAvg
```

## 🚀 Running the Code

To run a simulation on MNIST with 10 clients:

```bash
python main.py --dataset mnist --clients 10 --alpha 0.1 --rounds 140
```

Other options:

- `--dataset`: `mnist`, `fmnist`, or `cifar10`
- `--alpha`: Dirichlet concentration parameter (e.g., 0.1)
- `--peer-refinement`: enable/disable peer phase
- `--rounds`: total global rounds

## 📂 Repository Structure

```
├── main.py               # Main training script
├── models/               # Model architectures
├── utils/                # Helper functions and metrics
├── data/                 # Data loading and partitioning
├── results/              # Output logs and results
└── README.md             # This file
```

## 🧩 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{fernandes2025fedp2pavg,
  title={FedP2PAvg: A Peer-to-Peer Collaborative Framework for Federated Learning in Non-IID Scenarios},
  author={Fernandes, Bruno J. T. and Freire, Agostinho and de Andrade, João V. R. and Silva, Leandro H. S. and Navarro-Guerrero, Nicolás},
  booktitle={34th International Conference on Artificial Neural Networks (ICANN)},
  year={2025}
}
```

## 👥 Authors

- Bruno J. T. Fernandes (UPE, Leibniz Universität Hannover)
- Agostinho Freire (UPE)
- João V. R. de Andrade (UPE)
- Leandro H. S. Silva (UPE)
- Nicolás Navarro-Guerrero (Leibniz Universität Hannover)

## 📬 Contact

For questions or collaborations, feel free to reach out:

- Bruno Fernandes: `bruno.fernandes@upe.br`
- [Link to Paper (coming soon)](https://...)

---

*This work was supported by CAPES, FACEPE, CNPq, and the Alexander von Humboldt Foundation.*
