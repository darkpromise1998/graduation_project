import torch
import torchvision
import os
import json
import pickle
import argparse
from PIL import Image
import torch.utils.data as data
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import time
# from models import my_alexnet 
from dataset import adjust_learning_rate
from tensorboardX import SummaryWriter
from datetime import datetime

def train(args):
    
    save_dir = os.path.join(args.save_dir, args.model_pre + args.backbone + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(save_dir)
    writer = SummaryWriter(save_dir)
    
    multi_gpus = False  
    if len(args.gpus.split(',')) > 1:
        print('{} GPUs for use'.format(len(args.gpus.split(','))))
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.label_dict_path, 'rb') as f:
        label_dict = pickle.load(f)

    dataset = TrainData(args.root_path, label_dict, input_size=224)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

#     model = torchvision.models.resnet50()
#     model.fc = nn.Sequential(
#                              nn.Linear(2048, 512, bias=False),
#                              nn.BatchNorm1d(512),
#                              nn.ReLU(inplace=True),
#                              nn.Linear(512, args.feature_dim, bias=False))

    model = torchvision.models.alexnet()
    model.classifier[6] =  torch.nn.Linear(in_features=4096, out_features=args.feature_dim, bias=False)

#     model = my_alexnet(sobel=False, bn=True, out=args.feature_dim)
   
    args.lr_decay_epochs = [int(step) for step in args.lr_decay_epochs.split(',')]
    args.start_epoch = 1
    total_iters = 0
#     if args.resume and os.path.isfile(args.resume):

#         checkpoint = torch.load(args.resume, map_location='cpu')
#         args.start_epoch = checkpoint['epoch'] + 1
#         model.load_state_dict(checkpoint['model'])
#         total_iters = checkpoint['iters']
#         print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

#         del checkpoint
#         torch.cuda.empty_cache()
            
    if multi_gpus:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
           
    criterion = torch.nn.BCEWithLogitsLoss()
   
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9, nesterov=True, weight_decay=1e-5) 
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.init_lr, weight_decay=1e-4)
    model.train()
    
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        torch.cuda.empty_cache()
        
    len_dataloader = len(dataloader)

    for epoch in range(args.start_epoch, args.total_epoch + 1):

        adjust_learning_rate(epoch, args, optimizer)        
        print('Train Epoch: {}/{} ...'.format(epoch, args.total_epoch))
        
#         total_loss = []
        s = time.time()

        for step, (imgs, label) in enumerate(dataloader):

            imgs = imgs.to(device)
            labels = label.to(device)
            target = torch.sigmoid(labels)

            optimizer.zero_grad()
            output = model(imgs)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
#             total_loss.append(loss.item())
            total_iters += 1
            
            if (step + 1) % args.log_step == 0:
                duration = (time.time() - s) / args.log_step
                examples_per_sec = args.total_epoch / float(duration)
                print('Epoch: [%d/%d], Step: [%d/%d],  loss = %.4f,  %.2f examples/sec,  %.2f sec/batch' %
                      (epoch, args.total_epoch, step + 1, len_dataloader, loss.item(), examples_per_sec, duration))
                s = time.time()
                writer.add_scalar('loss', loss.item(), total_iters)
                writer.add_scalar('sup_lr', optimizer.param_groups[0]['lr'], total_iters)               
                writer.add_scalar('epoch', epoch, total_iters)

#         print('Speed: %.2f for one epoch %d/%d,  Mean Loss = %.4f' %
#               (time.time() - start_time, epoch, EPOCH, sum(total_loss) / len(total_loss)))
        
        if epoch % args.save_freq == 0:
            if multi_gpus:
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
                
            state = {
                'model': model.state_dict(),
#                 'optimizer': optimizer.state_dict(),
#                 'epoch': epoch,
#                 'iters': total_iters, 
            }
                
            torch.save(state, os.path.join(save_dir, 'Epoch_%02d_Iter_%06d_model.pth' % (epoch, total_iters)))
            del state
    
    print('Finishing training!')
    writer.close()

    
class TrainData(data.Dataset):

    def __init__(self, root_path, label_dict, input_size=224, transform=None):
        if transform is None:
            transform = transforms.Compose([transforms.RandomCrop(224),
                                            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
#                                             transforms.RandomGrayscale(p=0.2),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])

        self.root = root_path
        self.transform = transform
        self.label_dict = label_dict
        self.label_list = list(label_dict.keys())
        print(len(self.label_list))

    def __getitem__(self, index):
        fid = self.label_list[index]        
        label = self.label_dict[fid]
        fid = fid.split('.')[0]+'.jpg'
        
        img = Image.open(os.path.join(self.root, fid)).convert('RGB')
        img = self.transform(img)
        label = torch.Tensor(label)

        return img, label

    def __len__(self):
        return len(self.label_list)
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch for SSL on multimodal')
    
    parser.add_argument('--root_path', type=str, 
                        default='/share/fuzheren/MyDataset/InstaCities1M/img', 
                        help='train image root')
    parser.add_argument('--label_dict_path', type=str, default='../LDA/texts_label_doc2vec_ins.pk', help='label dict')
    
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--total_epoch', type=int, default=60, help='total epochs')
    parser.add_argument('--num_workers', type=int, default=12, help='num_workers for dataloader')
    parser.add_argument('--gpus', type=str, default='1', help='model prefix')
    parser.add_argument('--feature_dim', type=int, default=200, help='feature dimension for output')  
    
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--log_step', type=int, default=20, help='log_step for tensorboard')
    parser.add_argument('--save_dir', type=str, default='../training_result', help='model save dir')
    parser.add_argument('--model_pre', type=str, default='TextTopicNet_d2v_ins_', help=' model prefix ')
    parser.add_argument('--backbone', type=str, default='alexnet', help='backbone for feature learning')   

    parser.add_argument('--optimizer', type=str, default='SGD', help=' optimizer type ')
    parser.add_argument('--init_lr', type=float, default=0.01, help=' init lr for optimizer ')
    parser.add_argument('--lr_decay_epochs', type=str, default='20, 40', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
       
    parser.add_argument('--resume', 
                        default='', 
                        type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    
    args = parser.parse_args()
    train(args)
        