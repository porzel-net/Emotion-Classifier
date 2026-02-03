from torchvision import datasets, transforms
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm 
import time
import os


# --- PARAMETERS ---
RANDOM_SEED = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 40
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- MODEL ARCHITECTURE ---
def conv(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super().__init__()
        self.conv1 = conv(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.SiLU()
        self.conv2 = conv(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, layers, labels):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.SiLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=0.35)
        self.fc = nn.Linear(512, labels)

    def forward(self, x):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        return self.fc(out)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

# --- EVALUATION FUNCTIONS ---
def compute_accuracy(model, data_loader, device):
    model.eval()
    correct_pred, num_examples = 0, 0
    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            logits = model(features)
            predicted_labels = torch.argmax(logits, dim=1)
            correct_pred += (predicted_labels == targets).sum()
            num_examples += targets.size(0)
    return correct_pred.float() / num_examples * 100

def get_predictions(model, data_loader, device):
    model.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            
            
            logits = model(features)
            
            
            predictions = torch.argmax(logits, dim=1)
            
            
            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(predictions.cpu().numpy().tolist())
            
    return all_targets, all_preds

def hold_out_split (inputs, labels, test_size = 0.2):
    
    dataset_size = len(inputs)

    split = int (dataset_size * (1.0 - test_size))

    inputs_train = inputs[:split]
    labels_train = labels[:split]

    inputs_test = inputs[split:]
    labels_test = labels[split:]

    return inputs_train, inputs_test, labels_train, labels_test


def accuracy (true_labels, pred_labels):

    correct_count = 0.0

    total_samples = len(true_labels)

    for i in range (total_samples):

        if true_labels[i] == pred_labels[i]:
            correct_count = correct_count + 1.0

    return correct_count / total_samples


def macro_precision(true_labels, pred_labels, number_emotions):

    total_precision = 0.0

    for emotion in range (number_emotions):

        true_positives = 0.0
        false_positives = 0.0

        for i in range(len(true_labels)):
            if pred_labels[i] == emotion:
                if true_labels[i] == emotion:
                    true_positives = true_positives + 1.0
                else:
                    false_positives = false_positives + 1.0

        if true_positives + false_positives == 0.0:
            precision_emotion = 0.0
        else:
            precision_emotion = true_positives / (true_positives + false_positives)

        total_precision = total_precision + precision_emotion

    return total_precision / number_emotions

def macro_recall (true_labels, pred_labels, number_emotions):

    total_recall = 0.0

    for emotion in range (number_emotions):
        true_positives = 0.0
        false_negatives = 0.0

        for i in range (len (true_labels)):
            if true_labels[i] == emotion:
                if pred_labels[i] == emotion:
                    true_positives = true_positives + 1.0
                else:
                    false_negatives = false_negatives + 1.0

        if true_positives + false_negatives == 0.0:
            recall_classes = 0.0

        else:
            recall_classes = true_positives / (true_positives + false_negatives)
        total_recall = total_recall + recall_classes

    return total_recall / number_emotions

def macro_f1 (true_labels, pred_labels, number_emotions):

    total_f1 = 0.0

    for emotion in range (number_emotions):

        true_positives = 0.0
        false_positives = 0.0
        false_negatives = 0.0

        for i in range(len(true_labels)):
            if pred_labels[i] == emotion and true_labels[i] == emotion:
                true_positives = true_positives + 1.0
            elif pred_labels[i] == emotion and true_labels[i] != emotion:
                false_positives = false_positives + 1.0
            elif pred_labels[i] != emotion and true_labels[i] == emotion:
                false_negatives = false_negatives + 1.0

        if true_positives + false_positives == 0.0:
            precision  = 0.0
        else:
            precision = true_positives / (true_positives + false_positives)

        if true_positives + false_negatives == 0.0:
            recall = 0.0
        else:
            recall = true_positives / (true_positives + false_negatives)

        if precision + recall == 0.0:
            classes_f1 = 0.0
        else:
            classes_f1 = 2.0 * precision * recall / (precision + recall)

        total_f1 = total_f1 + classes_f1

    return total_f1 / number_emotions



# --- MAIN ---
def main():
    torch.manual_seed(RANDOM_SEED)

    custom_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)), 
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),       
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dir = 'train' 
    test_dir = 'test'

    train_dataset = datasets.ImageFolder(root=train_dir, transform=custom_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=custom_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    neconet_model = ResNet(Block, [2,2,2,2], NUM_CLASSES)
    neconet_model.to(DEVICE)
    
    optimizer = Adam(neconet_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=41, T_mult=1, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    print(f"Training starting... Device: {DEVICE}")
    
    for epoch in range(NUM_EPOCHS):
        neconet_model.train()
        running_loss = 0.0
        
        batch_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for targets, labels in batch_loop:
            targets, labels = targets.to(DEVICE), labels.to(DEVICE)

            # Forward
            logits = neconet_model(targets)
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_loop.set_postfix(loss=loss.item())

        
        scheduler.step()
        
        
        train_acc = compute_accuracy(neconet_model, train_loader, DEVICE)
        print(f"Epoch {epoch+1} Done | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")

    
    test_acc = compute_accuracy(neconet_model, test_loader, DEVICE)
    print(f"\nRESULT - Test Accuracy: {test_acc:.2f}%")

    y_true, y_pred = get_predictions(neconet_model, test_loader, DEVICE)
    final_prec = macro_precision(y_true, y_pred, NUM_CLASSES)
    final_rec = macro_recall(y_true, y_pred, NUM_CLASSES)
    final_f1 = macro_f1(y_true, y_pred, NUM_CLASSES)

    print(f"Precision: {final_prec:.4f}")
    print(f"Recall:    {final_rec:.4f}")
    print(f"F1 Score:  {final_f1:.4f}")

if __name__ == '__main__':
    main()
