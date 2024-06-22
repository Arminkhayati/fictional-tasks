import torch.nn as nn
import torch.nn.functional as F
import torch

# Made by Xinyu and modified by Dr.Li and Jingyan
class MultiTaskModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(MultiTaskModel, self).__init__()
        self.num_tasks = out_features
        self.input_features = in_features

        self.shared_layers = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.AlphaDropout(0.1),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.AlphaDropout(0.1),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.AlphaDropout(0.1)
        )
        # A Classic Multi Task Learning Framework
        self.task_layers = nn.ModuleList([nn.Linear(128, 1) for _ in range(out_features)])#after your papaer I decided to use MTL to solve this prob

    # This forward propagation logic defines the chain propagation of our idea
    def forward(self, x):
        shared_output = self.shared_layers(x)
        task_outputs = []
        for i, task_layer in enumerate(self.task_layers):
            if i == 0:
                task_output = torch.sigmoid(task_layer(shared_output))
            else:
                task_output = torch.sigmoid(task_layer(shared_output)) * task_outputs[-1]
            task_outputs.append(task_output)

        return task_outputs# Here I difined the S(x)

    # Modified by Jingyan
    def custom_loss(self, task_outputs, targets, masks):
        loss = 0
        for i, task_output in enumerate(task_outputs):
            task_target = targets[i]
            task_mask = masks[i]
            task_loss = F.binary_cross_entropy(task_output, task_target.float(), reduction='none')
            task_loss = task_loss * task_mask.float() # [2048, 1]
            # print('task_loss', task_loss.shape)
            loss += task_loss.sum() / task_mask.sum()
        return loss
