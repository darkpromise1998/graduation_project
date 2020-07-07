import torch.utils.data as data
import torch
import torchvision
import os
import json
import argparse
from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim
import time
import numpy as np
from datetime import datetime

class TrainData(data.Dataset):

    def __init__(self, root_path, label_dict, input_size=224, transform=None):
        if transform is None:
            transform = transforms.Compose([transforms.RandomCrop(224),
                                            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
#                                             transforms.RandomGrayscale(p=0.2),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])

        self.root = root_path
        self.transform = transform
        self.label_dict = label_dict
        self.label_list = list(label_dict.keys())

    def __getitem__(self, index):
        fid = self.label_list[index]
        label = self.label_dict[fid]

        img = Image.open(os.path.join(self.root, fid)).convert('RGB')
        img = self.transform(img)
        label = torch.Tensor(label)

        return img, label

    def __len__(self):
        return len(self.label_list)
    
    
class MyCrossEntropy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, label):
        
        output = torch.mul(output, -label)
        output = torch.sum(output) / label.size(0)
        
        return output
    
class MyClsHead(torch.nn.Module):
    def __init__(self, dim=2048, num_instances=20000, temperature=0.05):
        super().__init__()
        
        self.weight = torch.nn.Parameter(torch.Tensor(num_instances, dim))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.temperature = torch.nn.Parameter(torch.full([], temperature))        
        
    def forward(self, embeddings):
        
        norm_weight = nn.functional.normalize(self.weight, dim=1)
        prediction_logits = nn.functional.linear(embeddings, norm_weight)

        output = prediction_logits / self.temperature
        output = nn.functional.log_softmax(output, dim=1)
        
        return output  
    
    
def adjust_learning_rate(epoch, opt, optimizer):

    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.init_lr * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
   
