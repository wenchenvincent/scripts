'''
This script illustrates how to debug pytorch models:
    - Record activations and gradients
    - Check if there is any infinite element in tensor
    - Save tensors
'''
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import sys

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 10)
        self.fc1_out = None

    def forward(self, x):
        """FWD"""
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
############################################################################
###  Auxiliary functions for debugging
############################################################################

activations = {}
## Hook to record activation tensors in the forward pass
def get_activation(name):
    def forward_hook(module, input, output):
        activations[name] = output.detach()
    return forward_hook

gradients = {}
## Hook to record gradient tensors in the backward pass
def get_gradient(name):
    def backward_hook(module, grad_input, grad_output):
        ## grad_output and grad_input are tuples
        ## grad_input are gradients wrt inputs of the layer
        ## grad_output are gradients wrt outputs of the layer
        gradients[name] = grad_output[0].detach()
    return backward_hook

## Function to check if there is any inf or nan in the tensor
## Returns true if there is any inf or nan.
def any_infinite(tensor):
    return torch.any( torch.logical_not( torch.isfinite(tensor) ) ).item()


def train(args, model, device, train_loader, optimizer, epoch):
    """Training function."""
    model.train()
    ## Record activations in the forward pass
    model.fc1.register_forward_hook(get_activation('fc1'))
    model.dropout2.register_forward_hook(get_activation('dropout2'))
    model.fc2.register_forward_hook(get_activation('fc2'))
    model.fc3.register_forward_hook(get_activation('fc3'))
    ## Record gradients in the backward pass
    model.fc3.register_full_backward_hook(get_gradient('fc3'))
    model.fc2.register_full_backward_hook(get_gradient('fc2'))
    model.fc1.register_full_backward_hook(get_gradient('fc1'))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.autocast('cuda'):
            output = model(data)
        print('fc1.weight=', model.fc1.weight)
        print('fc1.bias=', model.fc1.bias)
        print('fc1.activation=', activations['fc1'])
        print('dropout2.activation=', activations['dropout2'])
        print('fc2.weight=', model.fc2.weight)
        print('fc2.bias=', model.fc2.bias)
        print('fc2.activation=', activations['fc2'])
        print('fc3.weight=', model.fc3.weight)
        print('fc3.bias=', model.fc3.bias)
        print('fc3.activation=', activations['fc3'])
        loss = F.nll_loss(output, target)
        loss.backward()
        print('fc3.gradient=', gradients['fc3'])
        print('fc3.weight.grad=', model.fc3.weight.grad)
        print('fc3.bias.grad=', model.fc3.bias.grad)
        print('fc2.gradient=', gradients['fc2'])
        print('fc2.weight.grad=', model.fc2.weight.grad)
        ### Check if there is any inf or nan in model.fc.weight.grad
        if any_infinite(model.fc2.weight.grad):
            print('Not finite')
            torch.save(gradients['fc2'], 'A.pt')
            torch.save(activations['dropout2'], 'B.pt')
            torch.save(model.fc2.weight.grad, 'D.pt')
            sys.exit(0)
        print('fc2.bias.grad=', model.fc2.bias.grad)
        print('fc1.gradient=', gradients['fc1'])
        print('fc1.weight.grad=', model.fc1.weight.grad)
        print('fc1.bias.grad=', model.fc1.bias.grad)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                f"Train Epoch: {epoch} "
                f"[{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\t"
                f"Loss: {loss.item():.6f}"
            )
            if args.dry_run:
                break


def test(model, device, test_loader):
    """Testing function."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.autocast('cuda'):
                output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        f"\nTest set: Average loss: {test_loss:.4f}, "
        f"Accuracy: {correct}/{len(test_loader.dataset)} "
        f"({100. * correct / len(test_loader.dataset):.0f}%)\n"
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()


    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("../data", train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST("../data", train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
