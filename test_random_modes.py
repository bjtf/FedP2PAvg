import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import pickle
import os
import torch.nn.functional as F
import random
import pandas as pd
from datetime import datetime
BATCH_SIZE = 512

class NetMnist(nn.Module):
    def __init__(self):
        super(NetMnist, self).__init__()
        # Layer 1
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # Conv1
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling
        # Layer 2
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # Conv2
        self.dropout2d = nn.Dropout2d()  # Dropout2d
        # Layer 3
        self.fc1 = nn.Linear(320, 50)  # FC1
        self.dropout = nn.Dropout()  # Dropout
        # Layer 4
        self.fc2 = nn.Linear(50, 10)  # FC2

    def forward(self, x):
        # Layer 1 operations
        x = self.conv1(x)
        x = self.pool(x)
        x = F.relu(x)
        # Layer 2 operations
        x = self.conv2(x)
        x = self.dropout2d(x)
        x = self.pool(x)
        x = F.relu(x)
        # Flatten
        x = x.view(-1, 320)  # Flatten the tensor
        # Layer 3 operations
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        # Layer 4 operation
        x = self.fc2(x)
        return x
    
class NetMnistFC(nn.Module):
    def __init__(self):
        super(NetMnist, self).__init__()
        # Input layer
        self.fc1 = nn.Linear(784, 50)  # Fully connected layer from 784 to 50 neurons
        # Hidden layer
        self.fc2 = nn.Linear(50, 10)   # Fully connected layer from 50 to 10 neurons

    def forward(self, x):
        # Flatten the image from (1, 28, 28) to (784)
        x = x.view(-1, 784)
        # Forward pass through the first fully connected layer and apply ReLU activation
        x = F.relu(self.fc1(x))
        # Forward pass through the second fully connected layer to produce output
        x = self.fc2(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class NetCifar100(nn.Module):
    def __init__(self):
        super(NetCifar100, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.linear = nn.Linear(256 * BasicBlock.expansion, 100)

    def _make_layer(self, block, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class NetCifar10(nn.Module):
    def __init__(self, num_classes=10):
        super(NetCifar10, self).__init__()
        
        
        self.n_cls = 10
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*5*5, 384) 
        self.fc2 = nn.Linear(384, 192) 
        self.fc3 = nn.Linear(192, self.n_cls)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

  
def load_dataset(dataset, n_particoes, dirichlet_alpha):
    train_dataset = None
    test_dataset = None
    train_loader = None
    test_loader = None
    idx = None
    if dataset == 'mnist':
        # Carregando o dataset MNIST
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('../data', train=False, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000)
    elif dataset == 'rotatedmnist':
        # Carregando o dataset MNIST
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('../data', train=False, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000)
    
    elif dataset == 'fashionmnist':
        # Carregando o dataset Fashion-MNIST
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.FashionMNIST('../data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('../data', train=False, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000)
    
    elif dataset == 'cifar100':
        # Carregando o dataset CIFAR-100
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000)
    elif dataset == 'cifar10':
        # Carregando o dataset CIFAR-10
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # ])
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])

        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000)
    
    if train_dataset:
        # Dividindo o conjunto de treinamento em (N_redes+1) partes iguais
        # idx = np.random.permutation(len(train_dataset))
        # idx = np.array_split(idx, n_particoes)
        
        # Dividindo o conjunto de treinamento seguindo a distribuição Dirichlet
        class_indices = [np.where(np.array(train_dataset.targets) == i)[0] for i in np.unique(train_dataset.targets)]
        idx = [[] for _ in range(n_particoes)]
        for indices in class_indices:
            # Distribuição de Dirichlet para cada classe
            proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, n_particoes))
            proportions = (len(indices) * proportions).astype(int).tolist()
            for i, p in enumerate(proportions):
                idx[i].extend(indices[:p])
                indices = indices[p:]

        # Embaralhar cada partição
        idx = [np.random.permutation(x) for x in idx]
        
        if dataset == 'rotatedmnist':
            # Aplicando rotação específica para cada cliente
            rotation_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135]
            assert len(rotation_angles) >= n_particoes, "Número de ângulos deve ser pelo menos igual ao número de partições."
        
            # Criar transformações rotacionadas para cada cliente
            rotated_datasets = []
            for client_idx, indices in enumerate(idx):
                angle = rotation_angles[client_idx]
                transform = transforms.Compose([
                    transforms.RandomRotation((angle, angle)),  # Rotação fixa para o cliente
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])
                # Criar um subconjunto com transformação rotacionada
                subset = Subset(train_dataset, indices)
                subset.dataset.transform = transform
                rotated_datasets.append(subset)
        
    return train_dataset, test_dataset, train_loader, test_loader, idx



def loadNetwork(dataset, device):
    net = None
    if dataset == 'mnist':          
        net = NetMnist().to(device)
    elif dataset == 'cifar100':
        net = NetCifar100().to(device)
    elif dataset == 'cifar10':
        net = NetCifar10().to(device)
    elif dataset == 'rotatedmnist':
        net = NetMnist().to(device)
    elif dataset == "fashionmnist":
        net = NetMnist().to(device)
    return net

def configure_nets(dataset, n_redes, device):
    from torchsummary import summary

    nets = []
    for i in range(n_redes):
        
        nets.append(loadNetwork(dataset, device))
        if i == 0:
            summary(nets[i])
    return nets

import matplotlib.pyplot as plt
import numpy as np

def plot_class_distribution(idx, train_dataset, dataset, n_redes, n_epocas, dirichlet_alpha, versao):
    """
    Plots a bar chart showing the class distribution across partitions.

    Args:
        idx (list of lists): Indices of the partitions.
        train_dataset (Dataset): Dataset containing the labels.
        num_classes (int): Number of classes (default: 10 for MNIST).
    """
    # Calculate the class distribution for each partition
    
    num_classes=10
    if dataset == 'CIFAR100':
        num_classes = 100
    
    class_distributions = []
    for partition_indices in idx:
        part_targets = None
        if isinstance(train_dataset.targets, torch.Tensor):
            part_targets = [train_dataset.targets[index].item() for index in partition_indices]
        else:  # Caso seja uma lista (exemplo: CIFAR10)
            part_targets = [train_dataset.targets[index] for index in partition_indices]
        class_counts = [part_targets.count(class_label) for class_label in range(num_classes)]
        class_distributions.append(class_counts)
    
    # Convert to numpy for easier handling
    class_distributions = np.array(class_distributions)

    # Configure the chart
    num_partitions = len(idx)
    partition_indices = range(num_partitions)
    bar_width = 0.8

    # Colors for each class
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    # Create stacked bars
    fig, ax = plt.subplots(figsize=(14, 7))
    bottom_values = np.zeros(num_partitions)
    for class_idx in range(num_classes):
        ax.bar(
            partition_indices,
            class_distributions[:, class_idx],
            bottom=bottom_values,
            color=colors[class_idx],
            label=f'Class {class_idx}',
            width=bar_width
        )
        bottom_values += class_distributions[:, class_idx]

    # Chart configurations
    ax.set_xlabel('Partition', fontsize=14)
    ax.set_ylabel('Number of Samples', fontsize=14)
    ax.set_title('Class Distribution Across Partitions', fontsize=16)
    ax.set_xticks(partition_indices)
    ax.set_xticklabels([f'Partition {i}' for i in partition_indices], rotation=45)
    ax.legend(title='Classes', fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Ensure output directory exists
    output_dir = "partitions"
    os.makedirs(output_dir, exist_ok=True)

    # Save the figure
    filename = f"{dataset}_train_dataset_config_of_{n_redes}_Epochs_{n_epocas}_{dirichlet_alpha}_dirichlet_{versao}_versao.png"
    filepath = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(filepath)
    print(f"Figure saved at: {filepath}")

    #plt.tight_layout()
    #plt.show()

# Example usage:
# plot_class_distribution(idx, train_dataset, num_classes=10)


def train_nets(nets, n_epocas, train_dataset, idx, device):
    # Exibir a distribuição de classes em todas as partições antes do treinamento
    #print("Distribuição de classes em todas as partições:\n")
    #for i, partition_indices in enumerate(idx):
        #part_targets = [train_dataset.targets[index].item() for index in partition_indices]
        #class_counts = {class_label: part_targets.count(class_label) for class_label in sorted(set(part_targets))}
        #print(f"Partição {i}:")
        #for class_label, count in class_counts.items():
            #print(f"  Classe {class_label}: {count} imagens")
        #print()
    
    for i, net in enumerate(nets):
        optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)
        criterion = nn.CrossEntropyLoss()
    
        # Carregando a parte do conjunto de treinamento
        #print(idx[i])
        part_loader = DataLoader(Subset(train_dataset, idx[i]), batch_size=BATCH_SIZE, shuffle=True)
    
        # Treinando a rede neural
        for epoch in range(1, n_epocas): 
            net.train()
            for batch_idx, (data, target) in enumerate(part_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = net(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        print(f"Net: {i}, Epoch: {epoch}, Loss: {loss.item()}")
    return nets


def inicializar_clone(dataset, device):
    # Instanciando a rede neural que será treinada para copiar as outras redes
    net_clone = loadNetwork(dataset, device)
    optimizer_clone = optim.SGD(net_clone.parameters(), lr=0.01, momentum=0.5)
    criterion_clone = nn.KLDivLoss(reduction='batchmean')
    return net_clone, optimizer_clone, criterion_clone

def evaluate_network(network, loader, device):
    """ Avalia a rede e retorna a taxa de erro e acurácia. """
    correct = 0
    total = 0
    loss_sum = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = network(data)
            loss = criterion(outputs, target)
            loss_sum += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    error_rate = 1 - (correct / total)
    accuracy = correct / total
    return loss_sum / len(loader), error_rate, accuracy

def generate_selection_distribution(
    n_redes, 
    criterion_type, 
    current_node, 
    class_distributions=None, 
    gradients=None
):
    """
    Gera uma distribuição de probabilidade para selecionar o nó parceiro.

    Args:
        n_redes (int): Número de redes.
        criterion_type (str): Critério de seleção ('random', 'similarity', 'performance', 'gradient_diversity').
        current_node (int): Índice do nó atual.
        class_distributions (list of list): Distribuições de classes de cada nó (para 'similarity').
        performances (list of float): Desempenhos de cada nó (para 'performance').
        gradients (list of Tensor): Gradientes de cada nó (para 'gradient_diversity').

    Returns:
        list: Distribuição de probabilidade (soma = 1.0).
    """
    scores = np.zeros(n_redes)

    if criterion_type == 'random':
        scores = np.ones(n_redes)  # Distribuição uniforme

    elif criterion_type == 'similarity' and class_distributions is not None:
        current_distribution = class_distributions[current_node]
        scores = [
            1 - np.sum(np.abs(current_distribution - class_distributions[i]))
            if i != current_node else 0
            for i in range(n_redes)
        ]

    elif criterion_type == 'gradient_diversity' and gradients is not None:
        current_gradient = gradients[current_node]
        scores = [
            np.linalg.norm((current_gradient - gradients[i]).cpu().numpy())
            if i != current_node else 0
            for i in range(n_redes)
        ]

    # Normaliza os scores para formar uma distribuição de probabilidade
    scores = np.maximum(scores, 0)  # Evita valores negativos
    probabilities = scores / np.sum(scores) if np.sum(scores) > 0 else np.ones(n_redes) / n_redes
    return probabilities

def train_random(nets, n_redes, n_epochs, train_dataset, idx, device, criterion_type="random", **kwargs):
    """
    Treina redes de forma federada aleatória ou com base em um critério.

    Args:
        nets (list of nn.Module): Redes neurais.
        n_redes (int): Número de redes.
        n_epochs (int): Número de épocas por iteração.
        train_dataset (Dataset): Conjunto de treinamento.
        idx (list): Índices de cada nó.
        device (torch.device): Dispositivo a ser usado.
        criterion_type (str): Critério de seleção ('random', 'similarity', etc.).
        kwargs: Argumentos adicionais para critérios específicos.

    Returns:
        list: Redes treinadas.
    """
    # Definindo otimizadores para cada rede
    optimizers = [] 
    criterion = nn.CrossEntropyLoss()
    for i in range(n_redes):
        optimizers.append(torch.optim.SGD(nets[i].parameters(), lr=0.01, momentum=0.9))

    epoch_losses = [0.0] * n_redes  # Lista para armazenar as perdas de cada rede

    # Pré-calcular distribuições de probabilidade
    selection_distributions = [
        generate_selection_distribution(
            n_redes, 
            criterion_type, 
            current_node=i, 
            class_distributions=kwargs.get('class_distributions'), 
            gradients=kwargs.get('gradients')
        )
        for i in range(n_redes)
    ]

    # Cada rede gera dados sintéticos e ensina a outra
    for i in range(n_redes):
        net1 = nets[i]
        optimizer1 = optimizers[i]
        net1.train()

        # Seleciona o parceiro com base na distribuição
        j = np.random.choice(range(n_redes), p=selection_distributions[i])

        # Configura o DataLoader
        part_loader = DataLoader(Subset(train_dataset, idx[j]), batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(n_epochs): 
            net1.train()
            for batch_idx, (data, target) in enumerate(part_loader):
                data, target = data.to(device), target.to(device)
                optimizer1.zero_grad()
                output = net1(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer1.step()

        epoch_losses[i] += loss.item()

    return nets


def train_ring(nets, n_redes, n_epochs, train_dataset, idx, device, steps, ring_order):
    # Definindo otimizadores para cada rede
    optimizers = []
    criterion = nn.CrossEntropyLoss()
    for i in range(n_redes):
        optimizers.append(torch.optim.SGD(nets[i].parameters(), lr=0.01, momentum=0.9))

    epoch_losses = [0.0] * n_redes  # Lista para armazenar as perdas de cada rede

    if not ring_order:
    # Construir um anel aleatório de nós
        ring_order = list(range(n_redes))
        random.shuffle(ring_order)

    # Treinamento para cada nó no anel
    for i in range(n_redes):
        current_node = ring_order[i]
        net_current = nets[current_node]
        optimizer_current = optimizers[current_node]
        net_current.train()

        # Determinar os nós cujos dados serão usados para o treinamento
        training_nodes = [
            ring_order[(i + step) % n_redes] for step in range(steps + 1)
        ]  # Inclui o nó atual e os próximos `steps` nós no anel

        # Treinar o nó atual com os dados dos nós selecionados
        for node in training_nodes:
            part_loader = DataLoader(Subset(train_dataset, idx[node]), batch_size=BATCH_SIZE, shuffle=True)
            for epoch in range(n_epochs):
                for batch_idx, (data, target) in enumerate(part_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer_current.zero_grad()
                    output = net_current(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer_current.step()

            # Acumular a perda para o nó atual
            epoch_losses[current_node] += loss.item()

    return nets


def combine(nets, net_clone):
    # Combinando os pesos das redes ao final do treinamento
    with torch.no_grad():
        average_weights = [torch.zeros_like(p) for p in nets[0].parameters()]
        for net in nets:
            for avg_p, p in zip(average_weights, net.parameters()):
                avg_p.add_(p.data)
        for avg_p in average_weights:
            avg_p.div_(len(nets))
        for p, avg_p in zip(net_clone.parameters(), average_weights):
            p.data.copy_(avg_p)
        # Atualizando todas as redes para serem clones da rede média
        #for net in nets:
        #    for p, avg_p in zip(net.parameters(), average_weights):
        #        p.data.copy_(avg_p)
        for net in nets:
            for p, p_clone in zip(net.parameters(), nets[0].parameters()):
                p.data.copy_(p_clone.data)
            for buffer, buffer_clone in zip(net.buffers(), nets[0].buffers()):
                buffer.copy_(buffer_clone)
    #eval_comite(nets, test_loader, n_redes, device)
    return nets, net_clone

def eval_clone(net_clone, test_loader, device):
    criterion = nn.CrossEntropyLoss()  
    net_clone.eval()
    test_loss_clone = 0
    correct_clone = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output_clone = net_clone(data)
            test_loss_clone += criterion(output_clone, target).item() 
            pred_clone = output_clone.argmax(dim=1, keepdim=True) 
            correct_clone += pred_clone.eq(target.view_as(pred_clone)).sum().item()
    
    test_loss_clone /= len(test_loader.dataset)
    
    print('\nTest set: Model Clone: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss_clone, correct_clone, len(test_loader.dataset),
        100. * correct_clone / len(test_loader.dataset)))
    
    return 100.0 * correct_clone / len(test_loader.dataset)
    
def eval_comite(nets, test_loader, n_redes, device):
    criterion = nn.CrossEntropyLoss()
    for net in nets:
        net.eval()
    
    test_losses = [0]*n_redes
    corrects = [0]*n_redes
    correct_avg = 0
    avg_loss = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # Comitê de redes
            outputs = [net(data) for net in nets]
            avg_output = sum(outputs) / len(outputs)
            avg_pred = avg_output.argmax(dim=1, keepdim=True)
            avg_loss += criterion(avg_output, target).item()
            correct_avg += avg_pred.eq(target.view_as(avg_pred)).sum().item()
    
            for i, net in enumerate(nets):
                output = net(data)
                test_losses[i] += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                corrects[i] += pred.eq(target.view_as(pred)).sum().item()
    
    avg_loss /= len(test_loader.dataset)
    for i in range(n_redes):
        test_losses[i] /= len(test_loader.dataset)
    
    print('Test set: Model Average: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        avg_loss, correct_avg, len(test_loader.dataset),
        100. * correct_avg / len(test_loader.dataset)))
    
    for i in range(n_redes):
        print('Test set: Model {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            i, test_losses[i], corrects[i], len(test_loader.dataset),
            100. * corrects[i] / len(test_loader.dataset)))
        
def save_nets(dataset, nets, idx, n_redes, n_epocas, dirichlet_alpha, epoca, mode, versao, save_idx, versao_idx):
    # Salvar os pesos de todas as redes treinadas com informações adicionais
    #for i, net in enumerate(nets):
    #    filename = f"nets/{dataset}_network_{i}_of_{n_redes}_Epochs_{n_epocas}_weights_{dirichlet_alpha}_dirichlet_{epoca}_epoca_{mode}_mode_{versao}_versao.pth"
    #    torch.save(net.state_dict(), filename)
    
    # Salvar a configuração da base de treino com informações adicionais
    if save_idx:
        config_filename = f"idx/{dataset}_train_dataset_config_of_{n_redes}_Epochs_{n_epocas}_{dirichlet_alpha}_dirichlet_{versao_idx}_versao.pkl"
        with open(config_filename, "wb") as f:
            pickle.dump(idx, f)
        
def load_nets(dataset, device, n_redes, n_epocas, dirichlet_alpha, epoca, mode, versao, load_idx, versao_idx):
    
    nets = []
    for i in range(n_redes):
        filename = f"nets/{dataset}_network_{i}_of_{n_redes}_Epochs_{n_epocas}_weights_{dirichlet_alpha}_dirichlet_{epoca}_epoca_{mode}_mode_{versao}_versao.pth"
        if os.path.exists(filename):
            net = loadNetwork(dataset, device)
            net.load_state_dict(torch.load(filename))
            nets.append(net)
    idx = []
    if load_idx:
        config_filename = f"idx/{dataset}_train_dataset_config_of_{n_redes}_Epochs_{n_epocas}_{dirichlet_alpha}_dirichlet_{versao_idx}_versao.pkl"
        if os.path.exists(config_filename):
            with open(config_filename, "rb") as f:
                idx = pickle.load(f)
    return nets, idx
    

# Parâmetro N_redes
N_redes = 10
N_epocas = 10 
dataset = 'cifar10'
dirichlet_alpha = 0.1
steps = 0
versao = 'a'
versao_idx = 'a'
mode = 'fashionmnist'
criterion_type = 'similarity' #similarity, gradient_diversity, random


import uuid
columns = ["dataset", "versao", "criterion_type", "steps","accs"]
data_to_save = []
version_names = [uuid.uuid4() for _ in range(5)]
print(version_names)
for dataset in ["cifar10"]:
    for versao in version_names:
        for criterion_type in ["random", "similarity", "gradient_diversity"]:
            for steps in [0]:
                if criterion_type != 'random' and steps != 0:
                    continue
                #if versao == 'd' and steps == 0 and criterion_type == 'similarity':
                #    continue
                if steps == 0:
                    mode = criterion_type
                elif steps >= 1:
                    mode = "ring-" + str(steps)
                data_to_save = []

                #print(f"Treinando na base {dataset}: {N_redes} redes e {N_epocas} epocas e alpha = {dirichlet_alpha} e versao = {versao} e versao_idx = {versao_idx} e mode = {mode}.")
                print(f"Treinando na base {dataset}: {N_redes} redes e {N_epocas} epocas e alpha = {dirichlet_alpha} e versao = {versao} e versao_idx = {versao_idx} e mode = {mode} e criterion_type = {criterion_type} e steps = {steps}.")
                # Definição do dispositivo a ser usado (GPU se disponível)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                train_dataset, test_dataset, train_loader, test_loader, idx = load_dataset(dataset, N_redes, dirichlet_alpha)
                nets, idx_tmp = load_nets(dataset, device, N_redes, N_epocas, dirichlet_alpha, 0, mode, versao, True, versao_idx)
                if idx_tmp:
                    idx = idx_tmp
                if len(nets) == 0:
                    #separar o treino em configuração da rede e treino e treinar um pouco cada rede com seu próprio subconjunto antes de cada nova interação
                    nets = configure_nets(dataset, N_redes, device)
                    #save_nets(dataset, nets, idx, N_redes, N_epocas, dirichlet_alpha, 0, mode, versao, True, versao_idx)
                net_clone, optimizer_clone, criterion_clone = inicializar_clone(dataset, device)
                plot_class_distribution(idx, train_dataset, dataset, N_redes, N_epocas, dirichlet_alpha, versao)

                # Inicializar variáveis adicionais para critérios personalizados
                class_distributions = [
                    np.bincount(
                        [train_dataset.targets[idx_val] for idx_val in idx_part],
                        minlength=10
                    )
                    for idx_part in idx
                ] if dataset == 'mnist' or dataset == 'cifar10' else None

                accs = []

                for i in range(1, 150):
                    print(f"Epoch: {i}")
                    nets_tmp, _ = load_nets(dataset, device, N_redes, N_epocas, dirichlet_alpha, i, mode, versao, False, versao_idx)
                    if len(nets_tmp) == N_redes:
                        nets = nets_tmp
                    else:
                        nets = train_nets(nets, N_epocas, train_dataset, idx, device)
                        
                        # Atualizar gradientes
                        gradients = []
                        for net in nets:
                            net_gradient = torch.cat([p.grad.view(-1) for p in net.parameters() if p.grad is not None])
                            gradients.append(net_gradient)
                        
                        if "ring" in mode:
                            nets = train_ring(nets, N_redes, N_epocas, train_dataset, idx, device, steps, [])
                        else:
                            print("aqui")
                            nets = train_random(nets, N_redes, N_epocas, train_dataset, idx, device, criterion_type=criterion_type,  
                                class_distributions=class_distributions,
                                gradients=gradients)
                        #save_nets(dataset, nets, idx, N_redes, N_epocas, dirichlet_alpha, i, mode, versao, False, versao_idx)
                        eval_comite(nets, test_loader, N_redes, device)
                    nets, net_clone = combine(nets, net_clone)
                    accuracy = eval_clone(net_clone, test_loader, device)
                    accs.append(accuracy)

                eval_comite(nets, test_loader, N_redes, device)
                eval_clone(net_clone, test_loader, device)
                data_to_save.append([dataset, versao, criterion_type, steps, accs])
                df_data = pd.DataFrame(data_to_save, columns=columns)
                now = datetime.now()
                df_data.to_csv(f"outputs_training_{now}_{dataset}_{versao}_{criterion_type}_{steps}.csv", index=False)



