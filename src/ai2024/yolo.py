import torch
from torch import nn

class DetectionNetwork(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fully_connected = nn.Sequential(
            nn.Linear(32 * 5 * 5, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes * 5),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.convolution(x)

        output = torch.flatten(output, 1)
        output = self.fully_connected(output)
        output = output.reshape(output.shape[0], self.num_classes, 5)

        return output


class LocalizationLoss(nn.Module):
    def __init__(self):
        super(LocalizationLoss, self).__init__()

    def forward(self, output, target):
        """
        :param output:
            Dimensions : (N, M, 7)
                (i, j, 0) indique le niveau de confiance entre 0 et 1 qu'un vrai objet est représenté par le vecteur (n, m, :)
                Si (i, j, 0) est plus grand qu'un seuil :
                    (i, j, 1) est la position x centrale normalisée de l'objet prédit j de l'image i
                    (i, j, 2) est la position y centrale normalisée de l'objet prédit j de l'image i
                    (i, j, 3) est la largeur normalisée et la hauteur normalisée de l'objet prédit j de l'image i
                    (i, j, 4) est le score pour la classe "anneau" de l'objet prédit j de l'image i
        :param target: Le tenseur cible pour la tâche de détection:
            Dimensions: (N, 3, 5)
                Si un 1 est présent à (i, j, 0), le vecteur (i, j, 0:5) représente un objet.
                Si un 0 est présent à (i, j, 0), le vecteur (i, j, 0:5) ne représente aucun objet.
                Si le vecteur représente un objet (i, j, :):
                    (i, j, 1) est la position x centrale normalisée de l'objet j de l'image i.
                    (i, j, 2) est la position y centrale normalisée de l'objet j de l'image i.
                    (i, j, 3) est la largeur normalisée et la hauteur normalisée de l'objet j de l'image i.
                    (i, j, 4) est l'indice de la classe de l'objet j de l'image i.
        """

        # À compléter
        bce_criterion = nn.BCELoss(reduction="sum")
        mse_criterion = nn.MSELoss(reduction="sum")

        target_obj = (target[:, :, 0] == 1).to(torch.float)
        target_no_obj = (target[:, :, 0] == 0).to(torch.float)
        target_xywh = target[:, :, 1:4]

        target_classes = torch.zeros((target.shape[0], 3, 3)).to(target.get_device())
        for i in range(target.shape[0]):
            for j in range(target.shape[1]):
                target_classes[i, j, int(target[i, j, -1])] = 1

        output_obj = output[:, :, 0]
        output_xywh = output[:, :, 1:4]
        output_classes = output[:, :, 4:]

        loss_xywh = mse_criterion(output_xywh, target_xywh)
        loss_with_obj = bce_criterion(output_obj, target_obj)
        loss_without_obj = bce_criterion(output_obj, target_no_obj)
        loss_classes = bce_criterion(output_classes, target_classes)

        return 5 * loss_xywh + loss_with_obj + 0.5 * loss_without_obj + 3 * loss_classes
        # return 2 * loss_xywh + loss_classes
