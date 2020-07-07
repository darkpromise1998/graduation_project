from __future__ import print_function
import sys,os
import glob
import re
import torch
import numpy as np
import torch.nn as nn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import gensim
from gensim import utils, corpora, models
from gensim.corpora.wikicorpus import filter_wiki
from collections import defaultdict
from PIL import Image
from PIL import ImageFile


filter_more = re.compile('(({\|)|(\|-)|(\|})|(\|)|(\!))(\s*\w+=((\".*?\")|([^ \t\n\r\f\v\|]+))\s*)+(({\|)|(\|-)|(\|})|(\|))?', re.UNICODE | re.DOTALL | re.MULTILINE) 


def preprocess(raw):
    # Initialize Tokenizer
    tokenizer = RegexpTokenizer(r'\w+')

    # Initialize Lemmatizer
    lemma = WordNetLemmatizer()
    
    # create English stop words list
    en_stop = get_stop_words('en')
    
    # Decode Wiki Markup entities and remove markup
    text = filter_wiki(raw)
    text = re.sub(filter_more, '', text)

    # clean and tokenize document string
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    
    # remove stop words from tokens
    tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    tokens = [lemma.lemmatize(i) for i in tokens]

    # remove non alphabetic characters
    tokens = [re.sub(r'[^a-z]', '', i) for i in tokens]
    
    # remove unigrams and bigrams
    tokens = [i for i in tokens if len(i)>2]
    
    return tokens


# Function to compute average precision for text retrieval given image as input
def get_AP_img2txt(sorted_scores, given_image, top_k):
        consider_top = sorted_scores[:top_k]
        top_text_classes = [GT_txt2img[i[0]][1] for i in consider_top]
        class_of_image = GT_img2txt[given_image][1]
        T = top_text_classes.count(class_of_image)
        R = top_k
        sum_term = 0
        for i in range(0,R):
                if top_text_classes[i] != class_of_image:
                        pass
                else:
                        p_r = top_text_classes[:i+1].count(class_of_image)
                        sum_term = sum_term + float(p_r/len(top_text_classes[:i+1]))
        if T == 0:
                return 0
        else:
                return float(sum_term/T)

# Function to compute average precision for image retrieval given text as input
def get_AP_txt2img(sorted_scores, given_text, top_k):
        consider_top = sorted_scores[:top_k]
        top_image_classes = [GT_img2txt[i[0]][1] for i in consider_top]
        class_of_text = GT_txt2img[given_text][1]
        T = top_image_classes.count(class_of_text)
        R = top_k
        sum_term = 0
        for i in range(0,R):
                if top_image_classes[i] != class_of_text:
                        pass
                else:
                        p_r = top_image_classes[:i+1].count(class_of_text)
                        sum_term = sum_term + float(p_r/len(top_image_classes[:i+1]))
        if T == 0:
                return 0
        else:
                return float(sum_term/T)

            
class VOC2007_dataset(torch.utils.data.Dataset):
    def __init__(self, voc_dir, split='train', transform=None):
        # Find the image sets
        image_set_dir = os.path.join(voc_dir, 'ImageSets', 'Main')
        image_sets = glob.glob(os.path.join(image_set_dir, '*_' + split + '.txt'))
        assert len(image_sets) == 20
        # Read the labels
        self.n_labels = len(image_sets)
        images = defaultdict(lambda:-np.ones(self.n_labels, dtype=np.uint8)) 
        for k, s in enumerate(sorted(image_sets)):
            for l in open(s, 'r'):
                name, lbl = l.strip().split()
                lbl = int(lbl)
                # Switch the ignore label and 0 label (in VOC -1: not present, 0: ignore)
                if lbl < 0:
                    lbl = 0
                elif lbl == 0:
                    lbl = 255
                images[os.path.join(voc_dir, 'JPEGImages', name + '.jpg')][k] = lbl
        self.images = [(k, images[k]) for k in images.keys()]
        self.transform = transform
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = Image.open(self.images[i][0])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.images[i][1]    
    

class RegLog(nn.Module):
    
    # Creates logistic regression on top of frozen features
    # For AlexNet, sizess = [9600, 9216, 9600, 9600,9216]
    # (6,4,3,3,2) and (3,0,0,1,1)
    # Maxpool or Avgpool
    def __init__(self, conv):
        super(RegLog, self).__init__()

        if conv==1:
            self.av_pool = nn.AvgPool2d(6, stride=6, padding=3)
            s = 9600
        elif conv==2:
            self.av_pool = nn.AvgPool2d(4, stride=4, padding=0)
            s = 9216
        elif conv==3:
            self.av_pool = nn.AvgPool2d(3, stride=3, padding=1)
            s = 9600
        elif conv==4:
            self.av_pool = nn.AvgPool2d(3, stride=3, padding=1)
            s = 9600
        elif conv==5:
            self.av_pool = nn.AvgPool2d(2, stride=2, padding=0)
            s = 9216
        else:
            self.av_pool = nn.Identity()

            
    def forward(self, x):
        x = self.av_pool(x)
        x = x.view(x.size(0), -1)
        return x

    
def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by 0.2 every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
            