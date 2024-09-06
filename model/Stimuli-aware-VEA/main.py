# This code is written by Jingyuan Yang @ XD

"""Train FI_8 with Pytorch"""

import torch
import argparse
import torchvision
import torchvision.transforms as transforms
import os
import random
from data_FI8 import FI_8
from models import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from adamw import AdamW
# from torch.autograd import Variable
import utils
import math
import numpy as np
from torchvision import models
from tensorboardX import SummaryWriter
from models.TL import Triplet
from models.CE_loss_softmax import CELoss_softmax
from models.CE_loss_weighed import CELoss_weighed

# Parameters
parser = argparse.ArgumentParser(description='PyTorch FI_8 CNN Training')
parser.add_argument('--img_path', type=str, default='/home/yjy/Dataset/FI/FI_image/')
parser.add_argument('--train_csv_file', type=str, default='/home/yjy/Dataset/FI/csv_new_new/annotations_train.csv')
parser.add_argument('--val_csv_file', type=str, default='/home/yjy/Dataset/FI/csv_new_new/annotations_val.csv')
parser.add_argument('--test_csv_file', type=str, default='/home/yjy/Dataset/FI/csv_new_new/annotations_test.csv')
# parser.add_argument('--train_csv_file', type=str, default='/home/yjy/Dataset/FI/csv/face_train.csv')
# parser.add_argument('--val_csv_file', type=str, default='/home/yjy/Dataset/FI/csv/face_val.csv')
# parser.add_argument('--test_csv_file', type=str, default='/home/yjy/Dataset/FI/csv/face_test.csv')
parser.add_argument('--sal_path', type=str, default='/home/yjy/Dataset/FI/saliency_object/')
# parser.add_argument('--rcnn_path', type=str, default='/home/yjy/Dataset/FI/FI_pDET_feats/')
parser.add_argument('--rcnn_path', type=str, default='/home/yjy/Dataset/FI/FI_image_DET/')
parser.add_argument('--face_path', type=str, default='/home/yjy/Dataset/FI/FI_face/')
parser.add_argument('--ckpt_path', type=str, default='/home/yjy/Code/faceNet/ckpts2')
parser.add_argument('--model', type=str, default='ResNet_50', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='FI_8', help='Dataset')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkposint')

parser.add_argument('--if_decay', default=1, type=int, help='decay lr every 5 epochs')
parser.add_argument('--decay', default=0.1, type=float, help='decay value every 5 epochs')
parser.add_argument('--lr_adam', default=1e-5, type=float, help='learning rate for adam|5e-4|1e-5|smaller')
parser.add_argument('--lr_sgd', default=1e-3, type=float, help='learning rate for sgd|1e-3|5e-4')
parser.add_argument('--wd', default=5e-5, type=float, help='weight decay for adam|1e-4|5e-5')
parser.add_argument('--optimizer', default='adamw', type=str, help='sgd|adam|adamw')
parser.add_argument('--gpu', default=0, type=int, help='0|1|2|3')

parser.add_argument('--seed', default=66, type=int, help='just a random seed')
opt = parser.parse_args()

# set gpu device
torch.cuda.set_device(opt.gpu)


# random seed
def set_seed(seed=opt.seed):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.benchmark = False                   # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True            # cudnn
    # os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(seed=opt.seed)

writer = SummaryWriter()

best_test_acc = 0
best_test_acc_epoch = 0
start_epoch = 0

learning_rate_decay_start = 5
learning_rate_decay_every = 5
learning_rate_decay_rate = opt.decay

total_epoch = 55

path = os.path.join(opt.dataset + '_' + opt.model)

# Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.Resize(480),  # resize the short side to 480, and resize the long side proportionally
    transforms.RandomCrop(448),  # different from resize, randomcrop will crop a square of 448*448, disproportionally
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

tface_transform = transforms.Compose([
    transforms.Resize(48), # resize the short side to 480, and resize the long side proportionally
    transforms.RandomCrop(44), # different from resize, randomcrop will crop a square of 448*448, disproportionally
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_val = transforms.Compose([
        transforms.Resize(480),
        transforms.RandomCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

vface_transform = transforms.Compose([
    transforms.Resize(48),
    transforms.RandomCrop(44),
    transforms.ToTensor()
])

transform_test = transform_val
teface_transform = vface_transform

trainset = FI_8(csv_file=opt.train_csv_file, root_dir=opt.img_path, face_dir=opt.face_path,
                rcnn_dir=opt.rcnn_path,
                transform=transform_train, face_transform=tface_transform)
valset = FI_8(csv_file=opt.val_csv_file, root_dir=opt.img_path, face_dir=opt.face_path,
                rcnn_dir=opt.rcnn_path,
              transform=transform_val, face_transform=vface_transform)
testset= FI_8(csv_file=opt.test_csv_file, root_dir=opt.img_path, face_dir=opt.face_path,
                rcnn_dir=opt.rcnn_path,
              transform=transform_test, face_transform=teface_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=8)
valloader = torch.utils.data.DataLoader(valset, batch_size=opt.batch_size, shuffle=False, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False, num_workers=8)

# Model
if opt.model == 'ResNet_50':
    base_model = models.resnet50(pretrained=True)
    net = model_resnet(base_model)
elif opt.model == 'ResNet_101':
    base_model = models.resnet101(pretrained=True)
    net = model_resnet(base_model)

param_num = 0
for param in net.parameters():
    param_num = param_num + int(np.prod(param.shape))

print('==> Trainable params: %.2f million' % (param_num / 1e6))
#print(np.prod(net.lstm.parameters()[1].shape))

if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    # assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('/home/yjy/Code/faceNet/ckpts1/epoch-3.pkl', map_location="cuda:0")
    # resnet_dict = net.state_dict()
    # pre_dict = {k: v for k, v in checkpoint.items() if k in resnet_dict}
    # resnet_dict.update(pre_dict)
    # net.load_state_dict(resnet_dict)

    net.load_state_dict(checkpoint)
    # best_test_acc = checkpoint['best_test_acc']
    # best_test_acc_epoch = checkpoint['best_test_acc_epoch']
    # start_epoch = checkpoint['best_test_acc_epoch'] + 1s
else:
    print('==> Building model..')

if torch.cuda.is_available():
    net.cuda()

CEloss = nn.CrossEntropyLoss()
MSEloss = nn.MSELoss()
Triploss = Triplet(measure='cosine', max_violation=True) #MARGIN
CEloss_weighed = CELoss_weighed()
CEloss_softmax = CELoss_softmax()

if torch.cuda.is_available():
    CEloss = CEloss.cuda()
    Triploss = Triploss.cuda()
    MSEloss = MSEloss.cuda()
    # CELoss_weighed = CELoss_weighed.cuda()
    # classify_loss = Floss.cuda()


if opt.optimizer == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=opt.lr_adam, weight_decay=opt.wd)
elif opt.optimizer == 'adamw':
    optimizer = AdamW(net.parameters(), lr=opt.lr_adam, weight_decay=opt.wd, amsgrad=False)
elif opt.optimizer == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=opt.lr_sgd, momentum=0.9, weight_decay=5e-4)


def l2norm(X):
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

# Training
def train(epoch):
    # set_seed(seed=opt.seed)
    print('\nEpoch: %d' % epoch)
    global train_acc
    train_loss = 0
    train_loss1 = 0
    train_loss2 = 0
    train_loss3 = 0
    correct = 0
    total = 0

    if opt.if_decay == 1:
        if epoch >= learning_rate_decay_start:
            frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every + 1 # round
            decay_factor = learning_rate_decay_rate ** frac # how many times we have this decay

            if opt.optimizer == 'adam':
                current_lr = opt.lr_adam
            elif opt.optimizer == 'adamw':
                current_lr = opt.lr_adam
            elif opt.optimizer == 'sgd':
                current_lr = opt.lr_sgd
            current_lr = current_lr * decay_factor # new learning rate
            for rr in range(len(optimizer.param_groups)):
                utils.set_lr(optimizer, current_lr, rr)  # set the decayed learning rate
        else:
            if opt.optimizer == 'adam':
                current_lr = opt.lr_adam
            elif opt.optimizer == 'adamw':
                current_lr = opt.lr_adam
            elif opt.optimizer == 'sgd':
                current_lr = opt.lr_sgd
        print('learning_rate: %s' % str(current_lr))

    for batch_idx, data in enumerate(trainloader):
        images = data['image']
        image_name = data['img_id']
        label_emo = data['label_emo']
        label_senti = data['label_senti']
        faces = data['face']
        # saliency = data['saliency']
        rcnn = data['rcnn']
        fmask = data['face_mask']

        if torch.cuda.is_available():
            images = images.cuda()
            label_emo = label_emo.cuda()
            label_senti = label_senti.cuda()
            faces = faces.cuda()
            # saliency = saliency.cuda()
            rcnn = rcnn.cuda()
            fmask=fmask.cuda()

        optimizer.zero_grad()

        net.train()
        emo, senti= net(images, faces, rcnn, fmask)
        # label_one_hot = torch.zeros(label_senti.shape[0], 2).cuda().scatter_(1, label_senti.view(-1, 1), 1)
        # fea = l2norm(fea)

        # loss1 = CEloss(emo, label_emo)
        # loss2 = CEloss(senti, lasbel_senti)
        loss1 = CEloss_softmax(emo, label_emo)
        loss2 = CEloss_softmax(senti, label_senti)

        # loss = loss1 * (1 + loss2)
        loss = loss1 + loss2

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, net.lstm.parameters()), 1.0) #1
        optimizer.step()

        # print(loss.item())
        train_loss1 += loss1.item()
        train_loss2 += loss2.item()
        train_loss += loss.item()

        # emo = F.softmax(emo, dim=1)
        _, predicted = torch.max(emo.data, 1)
        total += emo.size(0)
        correct += predicted.eq(label_emo.data).cpu().sum().numpy()

        utils.progress_bar(batch_idx, len(trainloader),
                           'Train_Loss1: %.3f Train_Loss2: %.3f Train_Loss: %.3f '
                           '| Train_Acc: %.3f%% (%d/%d)'
            % (train_loss1/(batch_idx+1), train_loss2/(batch_idx+1),
               train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    train_acc = 100.*correct/total

    writer.add_scalar('data/Train_Loss1', train_loss1 /(batch_idx+1), epoch)
    writer.add_scalar('data/Train_Loss2', train_loss2 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Train_Loss', train_loss / (batch_idx + 1), epoch)
    writer.add_scalar('data/Train_Acc', train_acc, epoch)
    print('==> Saving model...')
    torch.save(net.state_dict(), os.path.join(opt.ckpt_path, 'epoch-%d.pkl' % epoch))

# Validation
def val(epoch):
    # set_seed(seed=opt.seed)
    global val_acc
    global best_val_acc
    global best_val_acc_epoch
    val_loss1 = 0
    val_loss2 = 0
    val_loss3 = 0
    val_loss = 0
    correct = 0
    total = 0
    for batch_idx, data in enumerate(valloader):
        images = data['image']
        label_emo = data['label_emo']
        label_senti = data['label_senti']
        faces = data['face']
        # saliency = data['saliency']
        rcnn = data['rcnn']
        fmask = data['face_mask']

        if torch.cuda.is_available():
            images = images.cuda()
            label_emo = label_emo.cuda()
            label_senti = label_senti.cuda()
            faces = faces.cuda()
            # saliency = saliency.cuda()
            rcnn = rcnn.cuda()
            fmask = fmask.cuda()

        with torch.no_grad():
            net.eval()

            emo, senti = net(images, faces, rcnn, fmask)

            # label_one_hot = torch.zeros(label_senti.shape[0], 2).cuda().scatter_(1, label_senti.view(-1, 1), 1)
            # fea = l2norm(fea)

            # loss1 = CEloss(emo, label_emo)
            # loss2 = CEloss(senti, label_senti)
            loss1 = CEloss_softmax(emo, label_emo)
            loss2 = CEloss_softmax(senti, label_senti)

            # loss = loss1 * (1 + loss2)
            loss = loss1 + loss2

            val_loss1 += loss1.item()
            val_loss2 += loss2.item()
            val_loss += loss.item()

            # emo = F.softmax(emo, dim=1)
            _, predicted = torch.max(emo.data, 1)
            total += label_emo.size(0)
            correct += predicted.eq(label_emo.data).cpu().sum().numpy()

            utils.progress_bar(batch_idx, len(valloader),
                               'Val_Loss1: %.3f Val_Loss2: %.3f Val_Loss: %.3f '
                               '| Val_Acc: %.3f%% (%d/%d)'
                               % (val_loss1 / (batch_idx + 1), val_loss2 / (batch_idx + 1),
                                  val_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    val_acc = 100. * correct / total
    writer.add_scalar('data/Val_Loss1', val_loss1 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Val_Loss2', val_loss2 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Val_Loss', val_loss /(batch_idx+1), epoch)
    writer.add_scalar('data/Val_Acc', val_acc, epoch)

# Test
def test(epoch):
    # set_seed(seed=opt.seed)
    global test_acc
    global best_test_acc
    global best_test_acc_epoch
    test_loss1 = 0
    test_loss2 = 0
    test_loss3 = 0
    test_loss = 0
    correct = 0
    total = 0

    for batch_idx, data in enumerate(testloader):
        images = data['image']
        label_emo = data['label_emo']
        label_senti = data['label_senti']
        faces = data['face']
        # saliency = data['saliency']
        rcnn = data['rcnn']
        fmask = data['face_mask']

        if torch.cuda.is_available():
            images = images.cuda()
            label_emo = label_emo.cuda()
            label_senti = label_senti.cuda()
            faces = faces.cuda()
            # saliency = saliency.cuda()
            rcnn = rcnn.cuda()
            fmask = fmask.cuda()

        with torch.no_grad():
            net.eval()
            emo, senti = net(images, faces, rcnn, fmask)

            # label_one_hot = torch.zeros(label_senti.shape[0], 2).cuda().scatter_(1, label_senti.view(-1, 1), 1)
            # fea = l2norm(fea)

            # loss1 = CEloss(emo, label_emo)
            # loss2 = CEloss(senti, label_senti)
            loss1 = CEloss_softmax(emo, label_emo)
            loss2 = CEloss_softmax(senti, label_senti)

            # loss = loss1 * (1 + loss2)
            loss = loss1 + loss2

            test_loss1 += loss1.item()
            test_loss2 += loss2.item()
            test_loss += loss.item()

            # emo = F.softmax(emo, dim=1)
            _, predicted = torch.max(emo.data, 1)
            total += label_emo.size(0)
            correct += predicted.eq(label_emo.data).cpu().sum().numpy()

            utils.progress_bar(batch_idx, len(testloader),
                               'Test_Loss1: %.3f Test_Loss2: %.3f Test_Loss: %.3f '
                               '| Test_Acc: %.3f%% (%d/%d)'
                               % (test_loss1 / (batch_idx + 1), test_loss2 / (batch_idx + 1),
                                  test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    test_acc = 100. * correct / total
    writer.add_scalar('data/Test_Loss1', test_loss1 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Test_Loss2', test_loss2 / (batch_idx + 1), epoch)
    writer.add_scalar('data/Test_Loss', test_loss / (batch_idx + 1), epoch)
    writer.add_scalar('data/Test_Acc', test_acc, epoch)

    # Save checkpoint.
    if test_acc > best_test_acc:
        print('==> Finding best acc..')
        # print("best_test_acc: %0.3f" % test_acc)
        state = {
            'net': net.state_dict() if torch.cuda.is_available() else net,
            'acc': test_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'test_model.t7'))
        best_test_acc = test_acc
        best_test_acc_epoch = epoch

for epoch in range(start_epoch, total_epoch):
    # set_seed(seed=opt.seed)
    train(epoch)
    val(epoch)
    test(epoch)

    print("best_test_acc: %0.3f" % best_test_acc)
    print("best_test_acc_epoch: %d" % best_test_acc_epoch)