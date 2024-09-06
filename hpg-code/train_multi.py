from __future__ import print_function
from __future__ import division


import time
import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from monai.data import DataLoader, DistributedSampler
from time import time
from datetime import datetime
from datetime import timedelta
from PIL import ImageFile

def main():
    # If processed path not found, process EmoSet-118K (train/val/test set)
    # if not os.path.exists('/red/ruogu.fang/share/emotion_adversarial_attack/data/processed/EmoSet-118K'):
        
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data', default="/red/ruogu.fang/share/emotion_adversarial_attack/data/processed/EmoSet-118K/", type=str, help='Where is the directory for the data')
    parser.add_argument('--model', type=str, help='WSCNet, CAERSNet, PDANet, Stimuli_Aware_VEA')
    parser.add_argument('--log_dir', default= './logs', type=str)
    parser.add_argument('--random_state', default=42, type=int)
    parser.add_argument('--local-rank', type=int)

    args = parser.parse_args()
    data_directory = args.data
    random_state = args.random_state
    model_name = args.model

    dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=10))
    
    from config import WSCNet_Config, CAERSNet_Config, PDANet_Config, Stimuli_Aware_VEA_Config

    if model_name == 'WSCNet':
        config = WSCNet_Config()
    elif model_name == 'CAERSNet':
        config = CAERSNet_Config()
    elif model_name=='PDANet':
        config = PDANet_Config()
    elif model_name=='Stimuli_Aware_VEA':
        config = Stimuli_Aware_VEA_Config()
    model = config.model

    np.random.seed(random_state)
    random.seed(random_state)
    torch.manual_seed(random_state)
    os.environ["PYTHONHASHSEED"] = str(random_state)

    epochs = config.epoch

    TRAIN_DATA_PATH = os.path.join(data_directory, 'train')
    VAL_DATA_PATH = os.path.join(data_directory, 'val')
    TEST_DATA_PATH = os.path.join(data_directory, 'test')

    BATCH_SIZE = config.batch_size
    LEARNING_RATE = config.learning_rate

    TRANSFORM_IMG = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
    train_sampler = DistributedSampler(dataset=train_data, even_divisible=True, shuffle=True)
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, sampler=train_sampler)

    val_data = torchvision.datasets.ImageFolder(root=VAL_DATA_PATH, transform=TRANSFORM_IMG)
    val_sampler = DistributedSampler(dataset=val_data, even_divisible=True, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, sampler=val_sampler)

    test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
    test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)


    # printing the dataset size.
    if dist.get_rank() == 0:
        print(f'Training the model:{model_name}')

    device = torch.device(f"cuda:{args.local_rank}")
    torch.cuda.set_device(device)
    args.rank = dist.get_rank()
    args.log_dir = os.path.join(args.log_dir,
                           f"run_{os.environ['SLURM_JOB_NAME']}" + datetime.now().strftime("%m-%d-%Y-%H:%M:%S"))
    writer = SummaryWriter(log_dir=args.log_dir)
    if args.rank == 0:
        print(f"[{args.rank}] " + f"Writing Tensorboard logs to {args.log_dir}")
    
    model.to(device)
    model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
    
    weight = torch.Tensor([1 - 15681/94481, 1 - 8434/94481, 1 - 12022/94481, 1 - 13036/94481, 1 - 8467/94481, 1 - 15837/94481, 1 - 10786/94481, 1- 10218/94481]).to(device)
    criterion = nn.CrossEntropyLoss(weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_loss = np.inf
    best_model = None
    best_metric = -1
    epoch_loss_values = []
    best_metric_epoch = 0
    val_interval = 1
    val_loss = 0
    
    global_step = 0

    for epoch in range(epochs):  # loop over the dataset multiple times
        if dist.get_rank() == 0:
            print("-" * 10, flush=True)
            print(f"[{dist.get_rank()}] " + "-" * 10 + f" epoch {epoch + 1}/{epochs}")
            
        model.train()
        epoch_loss = 0.0
        step = 0
        train_sampler.set_epoch(epoch)
        
        for step, (inputs, labels) in enumerate(train_data_loader):
            # get the inputs; data is a list of [inputs, labels]
            tik = time()
            step += 1
            global_step +=1
            
            # put the input and label into device
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            epoch_loss += loss.item()
            print(f"[{args.rank}] " + f"train: " +
                  f"epoch {epoch}/{epochs+1}, " +
                  f"step_within_epoch {step}/{len(train_data_loader) - 1}, " +
                  f"loss: {loss.item():.2f}, " +
                  f"time: {(time() - tik):.2f}s")
        
        scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        if dist.get_rank() ==0:
            print(f"[{dist.get_rank()}]" + f"epoch {epoch + 1}, avg loss: {epoch_loss:.4f}")

        # validation mode
        if (epoch + 1) % val_interval == 0:
            model.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for val_step, val_data in enumerate(val_data_loader):
                    val_images, val_labels = (
                        val_data[0].to(device, non_blocking=True),
                        val_data[1].to(device, non_blocking=True),
                    )

                    outputs = model(val_images)
                    val_loss = criterion(outputs, val_labels)
                    epoch_val_loss += val_loss

                if epoch == 0:
                    best_loss = epoch_val_loss
                    best_model = model
                    best_metric_epoch = epoch

                    torch.save(best_model.module.state_dict(),
                                        os.path.join(args.log_dir,
                                                    model_name
                                                    + str(best_metric_epoch + 1)
                                                    + f"loss{epoch_loss:.2f}"
                                                + '.pth'))

                elif epoch != 0:
                    if epoch_val_loss < best_loss:  # val_loss < best_loss: # YY I think this should use best_metric instead of loss to save the best model
                        best_loss = epoch_val_loss
                        best_model = model
                        best_metric_epoch = epoch
                        if dist.get_rank() == 0:
                            print(f"best_loss={best_loss}, best model has been updated")

                            torch.save(best_model.module.state_dict(),
                                       os.path.join(args.log_dir,
                                                    model_name
                                                    + str(best_metric_epoch + 1)
                                                    + f"loss{epoch_loss:.2f}"
                                                    + '.pth'))
                if dist.get_rank() == 0:

                    print(
                        f"current epoch: {epoch + 1}, current MSE: {epoch_val_loss}",
                        f" best MSE: {best_loss}",
                        f" at epoch: {best_metric_epoch + 1}"
                    )

    best_model_wts = copy.deepcopy(best_model.state_dict())

    dist.destroy_process_group()
    print('Finished Training')
    
if __name__ == "__main__":
    main()