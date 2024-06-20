import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsampler.imbalanced import ImbalancedDatasetSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from datetime import datetime

from model.alexnet import Alexnet


def main():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.empty_cache()

    if not os.path.exists('./savedmodel'):
        os.makedirs('./savedmodel', exist_ok=True)

    train_dir = './data/processed/videoframe/train/'
    val_dir = './data/processed/videoframe/val/'
    test_dir = './data/processed/videoframe/test/'

    epochs = 100
    batch = 16
    lr = 0.0001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(

        [transforms.Resize(size=(224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
         ])

    train_dataset = datasets.ImageFolder(root=train_dir,
                                         transform=transform)

    val_dataset = datasets.ImageFolder(root=val_dir,
                                       transform=transform)

    test_dataset = datasets.ImageFolder(root=test_dir,
                                        transform=transform)

    print('Size of the training set:', str(len(train_dataset)))
    print('Size of the validation set:', str(len(val_dataset)))
    print('Size of the test set:', str(len(test_dataset)))
    print()

    train_loader = DataLoader(
        train_dataset, batch_size=batch, sampler=ImbalancedDatasetSampler(train_dataset))

    val_loader = DataLoader(
        val_dataset, batch_size=batch, shuffle=False)

    test_loader = DataLoader(
        test_dataset, batch_size=batch, shuffle=False)

    classes = tuple(os.listdir(train_dir))

    model = Alexnet(pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0
    best_vloss = 1_000_000.

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        with torch.no_grad():

            running_vloss = 0.0
            for i, data in enumerate(val_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                vloss = criterion(outputs, labels)
                running_vloss += vloss
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        avg_vloss = running_vloss / (i + 1)

        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}.pth'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), os.path.join('./savedmodel', model_path))

        epoch_number += 1
