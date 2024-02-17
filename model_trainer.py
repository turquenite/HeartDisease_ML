import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def train(network: torch.nn.Module, eval_loader: DataLoader, train_loader: DataLoader, num_epochs: int = 10, lr:float = 0.001, optimizer_type = torch.optim.Adam, loss_function = torch.nn.BCELoss(), plot_loss: bool = False):
    optimizer = optimizer_type(network.parameters(), lr)

    epoch_train_losses = list()
    epoch_eval_losses = list()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    network.to(device)

    for _ in tqdm(range(num_epochs), total=num_epochs, desc=f"Training in {num_epochs} Epochs"):
        network.train()

        minibatch_training_loss = list()
        minibatch_evaluation_loss = list()
        
        for batch in train_loader:
            network_input, insight_values, targets = batch
            network_input = network_input.to(device)
            targets = targets.to(device)
            
            # Get Model Output (forward pass)
            output = network(network_input)

            # Compute Loss
            loss = loss_function(output, targets)

            # Compute gradients (backward pass)
            loss.backward()  

            # Perform gradient descent update step
            optimizer.step()

            # Reset gradients
            optimizer.zero_grad()

            minibatch_training_loss.append(loss.item())

        epoch_train_losses.append(np.mean(minibatch_training_loss))

        network.eval()

        for batch in eval_loader:
            network_input, insight_values, targets = batch

            network_input = network_input.to(device)
            targets = targets.to(device)
            output = network(network_input)
            loss = loss_function(output, targets)
            minibatch_evaluation_loss.append(loss.item())

        epoch_eval_losses.append(np.mean(minibatch_evaluation_loss))

    if plot_loss:
        plot_losses(epoch_train_losses, epoch_eval_losses)
        
    return (epoch_train_losses, epoch_eval_losses)

def plot_losses(train_losses: list, eval_losses: list):
    plt.title("Model-Loss")
    plt.plot(range(len(train_losses)), train_losses, "orange", range(len(eval_losses)), eval_losses, "dodgerblue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Training Loss", "Evaluation Loss"])
    plt.show()