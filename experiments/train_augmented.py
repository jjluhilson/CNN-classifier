"""Train the earlier augmented Food-101 classifier for one epoch."""

import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid

assert sys.version_info >= (3, 10), "Python 3.10 or newer is required."

content_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AugmentedDataset(Dataset):
    def __init__(self, original_dataset, transform, duplicate_fraction=0.2):
        """
        Args:
            original_dataset: The original dataset (e.g., ImageFolder).
            transform: The data augmentation transform to apply.
            duplicate_fraction: Fraction of the dataset to duplicate (default: 0.2 or 20%).
        """
        self.original_dataset = original_dataset
        self.transform = transform
        self.duplicate_fraction = duplicate_fraction

        # Randomly select indices to duplicate
        self.duplicate_indices = random.sample(
            range(len(self.original_dataset)),
            int(len(self.original_dataset) * self.duplicate_fraction)
        )

    def __len__(self):
        # Total size of the augmented dataset
        return len(self.original_dataset) + len(self.duplicate_indices)

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            # Return original sample
            image, label = self.original_dataset[idx]
        else:
            # Return augmented duplicate
            duplicate_idx = self.duplicate_indices[idx - len(self.original_dataset)]
            image, label = self.original_dataset[duplicate_idx]
        if self.transform:
            image = self.transform(image)

        return image, label


def get_num_workers():
    suggested_workers = 0
    if hasattr(os, 'sched_getaffinity'):
        try:
            suggested_workers = len(os.sched_getaffinity(0))
        except Exception:
            pass
    if suggested_workers == 0:
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            suggested_workers = cpu_count
    num_workers = min(8, suggested_workers)
    return num_workers


def denorm(x):
    '''
    Function to reverse the normalization so that we can visualise the outputs
    '''
    x = unnormalize(x)
    x = x.view(x.size(0), 3, 256, 256)
    return x


def show(img):
    '''
    function to visualise tensors
    '''
    if torch.cuda.is_available():
        img = img.cpu()
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)).clip(0, 1))


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):

        super(ResidualBlock, self).__init__()

        self.left = nn.Sequential(nn.Conv2d(inchannel, outchannel, kernel_size=3,
                                         stride=stride, padding=1, bias=False),
                                  nn.BatchNorm2d(outchannel),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(outchannel, outchannel, kernel_size=3,
                                         stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(outchannel))

        self.shortcut = nn.Sequential()

        if stride != 1 or inchannel != outchannel:

            self.shortcut = nn.Sequential(nn.Conv2d(inchannel, outchannel,
                                                 kernel_size=1, stride=stride,
                                                 padding = 0, bias=False),
                                          nn.BatchNorm2d(outchannel) )

    def forward(self, x):

        out = self.left(x)

        out += self.shortcut(x)

        out = F.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, ResidualBlock, num_classes = 101):

        super(ResNet, self).__init__()

        self.inchannel = 16
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size = 3, stride = 1,
                                            padding = 1, bias = False),
                                  nn.BatchNorm2d(16),
                                  nn.ReLU())

        self.layer1 = self.make_layer(ResidualBlock, 16, 2, stride = 2)
        self.layer2 = self.make_layer(ResidualBlock, 32, 2, stride = 2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 2, stride = 2)
        self.layer4 = self.make_layer(ResidualBlock, 128, 2, stride = 2)
        self.layer5 = self.make_layer(ResidualBlock, 256, 2, stride = 2)
        self.layer6 = self.make_layer(ResidualBlock, 512, 2, stride = 2)
        self.maxpool = nn.MaxPool2d(4)
        self.fc = nn.Linear(512, num_classes)


    def make_layer(self, block, channels, num_blocks, stride):

        strides = [stride] + [1] * (num_blocks - 1)

        layers = []

        for stride in strides:

            layers.append(block(self.inchannel, channels, stride))

            self.inchannel = channels

        return nn.Sequential(*layers)


    def forward(self, x):

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def MyResNet():
    return ResNet(ResidualBlock)


def confusion(preds, y):
    """Plot a confusion matrix normalised by the true class."""
    with open(content_path + '/food-101/meta/labels.txt', 'r') as file:
        labels = [line.strip() for line in file if line.strip()]

    cm = confusion_matrix(
        y.cpu().numpy(), preds.cpu().numpy(),
        normalize='true', labels=range(len(labels)),
    )
    fig, ax = plt.subplots(1, 1, figsize=(55, 50))
    sns.heatmap(cm, annot=True, annot_kws={"size": 8}, ax=ax, cmap="Blues")

    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.xaxis.set_ticklabels(labels, rotation=90)
    ax.yaxis.set_ticklabels(labels, rotation=0)
    plt.show()


def incorrect_preds(preds, y, test_img):
    """Display up to nine misclassified images from a batch."""
    with open(content_path + '/food-101/meta/labels.txt', 'r') as file:
        labels = [line.strip() for line in file if line.strip()]

    correct = (preds == y).float()
    test_labels_check = correct.cpu().numpy()
    incorrect_indexes = np.where(test_labels_check == 0)

    test_img = test_img.cpu()
    samples = make_grid(
        denorm(test_img[incorrect_indexes][:9]), nrow=3,
        padding=2, normalize=False, value_range=None,
        scale_each=False, pad_value=0,
    )
    plt.figure(figsize=(20, 10))
    plt.title('Incorrectly Classified Instances')
    show(samples)
    labels = np.asarray(labels)
    print('Predicted label', labels[preds[incorrect_indexes].cpu().numpy()[:9]])
    print('True label', labels[y[incorrect_indexes].cpu().numpy()[:9]])
    print('Corresponding images are shown below')


def check_accuracy(loader, model, analysis=False, hotdog_data=False):
    """Evaluate top-1 accuracy and recall for hot dogs (class 55)."""
    num_correct = 0
    num_samples = 0
    target_class_correct = 0
    target_class_samples = 0
    target_class = 55
    model.eval()
    with torch.no_grad():
        for t, (x, y) in enumerate(loader):
            x = x.to(device=device, dtype=dtype)
            if hotdog_data:
                y = torch.tensor(target_class).repeat(x.shape[0]).to(
                    device=device, dtype=torch.long,
                )
            else:
                y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            mask = (y == target_class)
            if mask.sum() > 0:
                target_class_correct += (preds[mask] == y[mask]).sum()
                target_class_samples += mask.sum()

            if t == 0 and analysis:
                stack_labels = y
                stack_predicts = preds
            elif analysis:
                stack_labels = torch.cat([stack_labels, y], 0)
                stack_predicts = torch.cat([stack_predicts, preds], 0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f%%)' % (
            num_correct, num_samples, 100 * acc,
        ))

        if target_class_samples > 0:
            target_class_acc = float(target_class_correct) / target_class_samples
            print('Got %d / %d correct Hot Dogs (class %d) (%.2f%%)' % (
                target_class_correct, target_class_samples,
                target_class, 100 * target_class_acc,
            ))
        else:
            target_class_acc = 0.0
            print(f'No samples found for class {target_class}.')

        if analysis:
            confusion(stack_predicts, stack_labels)
            incorrect_preds(preds, y, x)
        return float(acc)


def train_part(model, optimizer, epochs=1):
    """
    Train a model on food101 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            loss.backward()

            # Update the parameters of the model using the gradients
            optimizer.step()

            if t % print_every == 0:
                print('Epoch: %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
        check_accuracy(loader_val, model)


if __name__ == "__main__":
    rng_seed = 90
    torch.manual_seed(rng_seed)

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist()),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist()),
        ]
    )
    train_path = content_path+'/food-101/train'
    test_path = content_path+'/food-101/test'

    original_train_dataset = datasets.ImageFolder(train_path)
    test_dataset = datasets.ImageFolder(test_path, transform=test_transform)

    augmented_train_dataset = AugmentedDataset(original_train_dataset, transform=train_transform, duplicate_fraction=0.2)

    # Split the augmented dataset into training and validation subsets.
    n = len(augmented_train_dataset)
    n_val = int(n/10)

    train_set, val_set = torch.utils.data.random_split(augmented_train_dataset, [n-n_val, n_val])

    print(len(train_set), len(val_set), len(test_dataset))

    batch_size = 32

    num_workers = get_num_workers()
    print(f"Using {num_workers} workers")
    loader_train = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    loader_val = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    loader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

    USE_GPU = True
    dtype = torch.float32

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(device)

    print_every = 10

    # Train the classifier.
    model = MyResNet()
    optimizer = optim.Adamax(model.parameters(), lr=0.0001, weight_decay=1e-7)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params))

    train_part(model, optimizer, epochs=1)

    # Report validation accuracy
    check_accuracy(loader_val, model, analysis=True)

    # Save the model weights.
    torch.save(model.state_dict(), os.path.join(content_path, 'checkpoints', 'augmented_1_epoch.pt'))
