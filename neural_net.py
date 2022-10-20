import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

from collections import OrderedDict

import numpy as np
import pandas as pd


def prediction(image):
    PATH = "CNN.pt"

    classes = ['A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
        'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'Too Many Hands', 'U',
        'V', 'W', 'X', 'Y', 'Z']

    model = models.resnext50_32x4d(weights=None)
    model.fc = nn.Linear(2048, 28)
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.eval()

    transform = transforms.Compose([transforms.Resize((256, 256)),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    batch_t = torch.unsqueeze(transform(image), 0)
    output = model(batch_t)

    activation_func = nn.functional.softmax(output, dim=1)
    prob, pred = torch.topk(activation_func, k = 3, dim = 1)
    prob, pred = prob.detach().numpy(), pred.detach().numpy()
 
    return [classes[i] for i in pred[0]], [p * 100 for p in prob[0]]