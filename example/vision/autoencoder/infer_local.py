"""
Example for using `coeus` version context manager.
"""
import os
import time
import torch
from torch.jit import script
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms, utils


batch_size = 64

data_folder = Path("/home/data/MNIST/data")
results_folder = Path("/home/data/results")

# Train data
X_train = datasets.MNIST(
    root=data_folder,
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    dataset=X_train,
    batch_size=batch_size,
    shuffle=True
)

# Test data
X_test = datasets.MNIST(
    root=data_folder,
    train=False,
    transform=transforms.ToTensor()
)

test_loader = torch.utils.data.DataLoader(
    dataset=X_test,
    batch_size=batch_size,
    shuffle=False
)


device = 'cuda'
model = torch.jit.load(os.path.join(results_folder, "autoencoder.pt"))


def test_batch():
    model.eval()

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)

            print(data.shape)
            # pylint: disable-next=not-callable
            output = model(data)
            print(output)
            print(output.shape)

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([
                    data[:n],
                    output.view(batch_size, 1, 28, 28)[:n]
                ])
                path = os.path.join(results_folder, f"reconstruction_reloaded.png")
                utils.save_image(comparison.cpu(), path, nrow=n)


if __name__ == "__main__":
    test_batch()
