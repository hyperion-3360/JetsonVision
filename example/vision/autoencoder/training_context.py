"""
Example for using `coeus` version context manager.
"""
import os
from pathlib import Path

import torch
from torch.jit import script
import matplotlib.pyplot as plt

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


class Autoencoder(torch.nn.Module):
    """
    Small Autoencoder for dimensionality reduction
    """
    def __init__(self) -> None:
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 28, kernel_size=3, stride=1, padding=1),
            torch.nn.PReLU(),
            torch.nn.Conv2d(28, 28, kernel_size=3, stride=1, padding=1),
            torch.nn.PReLU(),
            torch.nn.Conv2d(28, 28, kernel_size=3, stride=2, padding=1),
            torch.nn.PReLU(),
            torch.nn.Conv2d(28, 1, kernel_size=3, stride=1, padding=1)
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(1, 28, kernel_size=3, stride=1, padding=1),
            torch.nn.PReLU(),
            torch.nn.ConvTranspose2d(28, 28, kernel_size=3, stride=1, padding=1),
            torch.nn.PReLU(),
            torch.nn.ConvTranspose2d(28, 28, kernel_size=4, stride=2, padding=1),
            torch.nn.PReLU(),
            torch.nn.ConvTranspose2d(28, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
print(f"DEVICE: {device}")
model = Autoencoder().to(device)


criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-8
)


def train_batch(epoch):
    model.train()

    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()

        # pylint: disable-next=not-callable
        output = model(data)

        loss = criterion(output, data)
        loss.backward()
        train_loss += loss.item()

        optimizer.step()

        if batch_idx % 10 == 0:
            current = batch_idx * len(data)
            total =  len(train_loader.dataset)
            percent = 100. * batch_idx / len(train_loader)
            batch_loss = loss.item() / len(data)
            print(
                f"Train Epoch: {epoch} [{current}/{total} ({percent:.0f}%)]"
                f"    Loss: {batch_loss:.6f}"
            , end="\r")

    train_loss /= len(train_loader.dataset)
    print(f"====> Epoch: {epoch} Average loss: {train_loss:.6f}                 ")
    return train_loss


def test_batch(epoch):
    model.eval()

    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)

            # pylint: disable-next=not-callable
            output = model(data)

            loss = criterion(output, data)
            test_loss += loss.item()

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([
                    data[:n],
                    output.view(batch_size, 1, 28, 28)[:n]
                ])
                path = Path(os.path.join(results_folder, f"reconstruction_{epoch}.png"))
                if not path.parent.is_dir():
                    path.parent.mkdir()

                utils.save_image(comparison.cpu(), path, nrow=n)

        test_loss /= len(test_loader.dataset)
        print(f"====> Test set loss: {test_loss:.6f}")
        return test_loss


def save_sample(epoch):
    with torch.no_grad():
        for data, _ in test_loader:
            data = data[0].to(device)

            # Original
            plt.subplot(1, 3, 1)
            original = data.cpu().numpy().reshape(28, 28)
            plt.title("Original")
            plt.imshow(original)
            plt.gray()

            # Code
            plt.subplot(1, 3, 2)
            encoded = model.encoder(data)
            plt.title("Encoded")
            plt.imshow(encoded.cpu().numpy().reshape(14,14))
            plt.gray()

            # Decoded
            plt.subplot(1, 3, 3)
            decoded = model.decoder(encoded).cpu().numpy().reshape(28, 28)
            plt.title("Decoded")
            plt.imshow(decoded)
            plt.gray()

            break

        plt.savefig(os.path.join(results_folder, f"sample_{epoch}.png"))


def train(epochs=10):
    for epoch in range(1, epochs + 1):
        train_loss = train_batch(epoch)
        test_lost = test_batch(epoch)
    save_sample(epoch)

    return epoch, train_loss, test_lost


def main():
    train()

    results_folder.mkdir(parents=True, exist_ok=True)
    
    torch.save(model, results_folder/"autoencoder.pkl")
    torch.jit.save(script(model), results_folder/"autoencoder.pt")


if __name__ == "__main__":
    main()
