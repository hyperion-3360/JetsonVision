# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import cv2
from pycocotools.coco import COCO

from pathlib import Path

# class DetectionNetwork(nn.Module): #YOLO
#     def __init__(self, num_classes=3, num_bbox=3):
#         self.num_classes = 2
#         self.num_bbox = num_bbox
#         self.lambda_coord = 5
#         self.lambda_noobj = 0.5
#         super(DetectionNetwork, self).__init__()

#         self.features = nn.Sequential( # input is 1x448x448
#             nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), # output: 64x224x224
#             nn.LeakyReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 64x112x112

#             nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1), # output: 192x112x112
#             nn.LeakyReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),  # 192x56x56

#             nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0), # output: 128x56X56
#             nn.LeakyReLU(),

#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # output: 256x56x56
#             nn.LeakyReLU(),
#             nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0), # output: 256x56x56
#             nn.LeakyReLU(),
#             nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0), # output: 128x56x56
#             nn.LeakyReLU(),


#             nn.MaxPool2d(kernel_size=2, stride=2),  # 256x28x28


#         )

#         self.reset_parameters()

#     def reset_parameters(self):
#         for layer in self.bbox_regressor:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_uniform_(layer.weight)
#                 nn.init.constant_(layer.bias, 0)

#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         bbox_output = self.bbox_regressor(x)
#         bbox_output = bbox_output.view(x.size(0), 3, 5)
#         #sigmoid_tensor = torch.cat([torch.sigmoid(bbox_output[:, :, :4]), bbox_output[:, :, 4:]], dim=2)
#         return bbox_output

#     def getLost(self, output, target):

#         # output shape: (batch_size, num_boxes, 5)
#         # target shape: (batch_size, num_boxes, 5)

#         obj_mask = target[:, :, 0] >= 0.5  # object is present in the cell
#         noobj_mask = target[:, :, 0] < 0.5  # object is not present in the cell

#         mse = nn.MSELoss()
#         # Compute the localization loss for boxes where an object is present
#         loc_loss = torch.sum(obj_mask * mse(output[:,:,1:4], target[:,:,1:4]))
#         loc_loss *= self.lambda_coord

#         # Compute the confidence loss for boxes where an object is present
#         conf_loss_obj = torch.sum(obj_mask * F.binary_cross_entropy(output[:, :, 0], target[:, :, 0], reduction='none'))
#         conf_loss_noobj = torch.sum(noobj_mask * F.binary_cross_entropy(output[:, :, 0], target[:, :, 0], reduction='none'))
#         conf_loss = conf_loss_obj + self.lambda_noobj * conf_loss_noobj

#         # Compute the classification loss for boxes where an object is present
#         class_loss = torch.sum(obj_mask * mse(output[:, :, 4:]*2.99, target[:, :, 4:]))

#         # Compute the total loss
#         loss = loc_loss + conf_loss + class_loss

#         return loss