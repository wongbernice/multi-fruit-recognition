from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from ConvNet import ConvNet 
from MultiFruitDataset import MultiFruitDataset
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, hamming_loss
import textwrap

def train(model, device, train_loader, optimizer, criterion, epoch, batch_size):
    '''
    Trains the model for an epoch and optimizes it.
    
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    train_loader: dataloader for training samples.
    optimizer: optimizer to use for model parameter updates.
    criterion: used to compute loss for prediction and target 
    epoch: Current epoch to train for.
    batch_size: Batch size to be used.
    '''
    
    # Set model to train mode before each epoch
    model.train()
    
    # Empty list to store losses 
    losses = []
    correct = 0
    
    # Iterate over entire training samples (1 epoch)
    for batch_idx, batch_sample in enumerate(train_loader):
        data, target = batch_sample
        
        # Push data/label to correct device
        data, target = data.to(device), target.to(device)
        
        # Reset optimizer gradients. Avoids grad accumulation (accumulation used in RNN).
        optimizer.zero_grad()
        
        # Do forward pass for current set of data
        output = model(data)
        
        # ======================================================================
        # Compute loss based on criterion
        loss = criterion(output, target)
        
        # Computes gradient based on final loss
        loss.backward()
        
        # Store loss
        losses.append(loss.item())
        
        # Optimize model parameters based on learning rate and gradient 
        optimizer.step()
        
        # Apply threshold to get predicted label (0 or 1)
        pred = (torch.sigmoid(output) > 0.5).float()
        
        # ======================================================================
        # Count correct predictions overall 
        correct += (pred == target).sum().item()
        
    train_loss = float(np.mean(losses))
    total_labels = sum([t.numel() for _, t in train_loader.dataset])
    train_acc = 100. * correct / total_labels
    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        float(np.mean(losses)), correct, total_labels,
        100. * correct / total_labels))
    return train_loss, train_acc
    


def test(model, device, test_loader, criterion):
    '''
    Tests the model.
    
    model: The model to train. Should already be in correct device.
    device: 'cuda' or 'cpu'.
    test_loader: dataloader for test samples.
    '''
    
    # Set model to eval mode to notify all layers.
    model.eval()
    
    losses = []
    correct = 0

    all_targets = []
    all_preds = []

    # Set torch.no_grad() to disable gradient computation and backpropagation
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            data, target = sample
            data, target = data.to(device), target.to(device)
            

            # Predict for data by doing forward pass
            output = model(data)
            
            # ======================================================================
            # Compute loss based on same criterion as training
            loss = criterion(output, target)
            
            # Append loss to overall test loss
            losses.append(loss.item())
            
            # Apply threshold to get predicted label (0 or 1)
            pred = (torch.sigmoid(output) > 0.5).float()
                        
            # ======================================================================
            # Count correct predictions overall 
            correct += (pred == target).sum().item()


            all_targets.append(target.cpu())
            all_preds.append(pred.cpu())
    
    all_targets = torch.cat(all_targets)
    all_preds = torch.cat(all_preds)

    test_loss = float(np.mean(losses))
    acc = 100. * (all_preds == all_targets).sum().item() / all_targets.numel()
    f1_micro = f1_score(all_targets, all_preds, average='micro')
    f1_macro = f1_score(all_targets, all_preds, average='macro')
    hamming = hamming_loss(all_targets, all_preds)

    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {acc:.2f}%, F1_micro: {f1_micro:.3f}, F1_macro: {f1_macro:.3f}, Hamming Loss: {hamming:.3f}')
    
    return test_loss, acc, f1_micro, f1_macro, hamming

def show_predictions(model, device, test_dataset, classes, n=6):
    """
    Show n random predictions from the test dataset.

    model: trained PyTorch model
    device: "cuda" or "cpu"
    test_dataset: multi-fruit test dataset
    classes: list of class names
    n: number of random images to display
    """

    model.eval()
    plt.figure(figsize=(5 * n, 5))

    for i in range(n):
        idx = np.random.randint(0, len(test_dataset))
        img, label = test_dataset[idx]

        input_img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_img)
            probs = torch.sigmoid(output).cpu()[0]  # probabilities per class

        img_disp = img.permute(1, 2, 0) * 0.5 + 0.5

        # True labels
        true_labels = [classes[j] for j in range(len(classes)) if int(label[j].item()) == 1]
        num_labels = len(true_labels)

        # Predicted labels
        topk = torch.topk(probs, num_labels)
        topk_indices = topk.indices.tolist()
        topk_probs = topk.values.tolist()
        pred_labels_with_probs = [f"{classes[topk_indices[j]]} ({topk_probs[j]*100:.1f}%)"
                                  for j in range(num_labels)]

        # Wrap text
        wrapped_true = "\n".join(textwrap.wrap(", ".join(true_labels), width=20))
        wrapped_pred = "\n".join(textwrap.wrap(", ".join(pred_labels_with_probs), width=25))

        plt.subplot(1, n, i + 1)
        plt.imshow(img_disp)
        plt.axis("off")
        plt.title(f"True: {wrapped_true}\nPred: {wrapped_pred}", fontsize=10)

        print(f"\nRandom test image index: {idx}")
        print("True labels:", true_labels)
        print("Predicted labels (top-k):", [classes[j] for j in topk_indices])
        print("Top-k class probabilities:")
        for j in range(num_labels):
            print(f"{classes[topk_indices[j]]}: {topk_probs[j]*100:.1f}%")

    plt.tight_layout()
    plt.subplots_adjust(top=0.8)
    plt.savefig("sample_predictions.png")
    plt.show()
    plt.close()

def main(FLAGS):
    # Check if cuda is available
    use_cuda = torch.cuda.is_available()
    
    # Set proper device based on cuda availability 
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Torch device selected: ", device)
    
    # Initialize the model and send to device 
    model = ConvNet().to(device)

    train_losses_list = []
    train_accuracies_list = []
    test_losses_list = []
    test_accuracies_list = []
        
    # ======================================================================
    # Define loss function.
    criterion = nn.BCEWithLogitsLoss()
    
    # ======================================================================
    # Define optimizer function.
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate, momentum=0.9)

    # Create transformations to apply to each data sample 
    # Can specify variations such as image flip, color flip, random crop, ...
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # Load datasets for training and testing
    # Inbuilt datasets available in torchvision (check documentation online)
    dataset1 = MultiFruitDataset(
        image_folder="../multi-fruit-dataset/Training",
        label_file="../multi-fruit-dataset/Training/labels_file.txt",
        transform=transform
    )
    dataset2 = MultiFruitDataset(
        image_folder="../multi-fruit-dataset/Testing",
        label_file="../multi-fruit-dataset/Testing/labels_file.txt",
        transform=transform
    )
    train_loader = DataLoader(dataset1, batch_size = FLAGS.batch_size, 
                                shuffle=True, num_workers=4)
    test_loader = DataLoader(dataset2, batch_size = FLAGS.batch_size, 
                                shuffle=False, num_workers=4)
    
    best_accuracy = 0.0
    best_f1_micro = 0.0
    best_f1_macro = 0.0
    best_hamming = float('inf')
    
    # Run training for n_epochs specified in config 
    for epoch in range(1, FLAGS.num_epochs + 1):
        print(f"\nEpoch {epoch}\n")
        train_loss, train_accuracy = train(model, device, train_loader,
                                            optimizer, criterion, epoch, FLAGS.batch_size)
        test_loss, test_accuracy, f1_micro, f1_macro, hamming = test(model, device, test_loader, criterion)
        
        # Store losses and accuracies
        train_losses_list.append(train_loss)
        train_accuracies_list.append(train_accuracy)
        test_losses_list.append(test_loss)
        test_accuracies_list.append(test_accuracy)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
        if f1_micro > best_f1_micro:
            best_f1_micro = f1_micro
        if f1_macro > best_f1_macro:
            best_f1_macro = f1_macro
        if hamming < best_hamming:
            best_hamming = hamming
    
    best_hamming_accuracy = 1 - best_hamming
    
    print("Best Accuracy: {:2.2f}".format(best_accuracy))
    print(f"Best F1 Micro: {best_f1_micro:.3f}")
    print(f"Best F1 Macro: {best_f1_macro:.3f}")
    print(f"Lowest Hamming Loss: {best_hamming:.3f}")
    print(f"Best Hamming Accuracy: {best_hamming_accuracy:.3f}")
    
    print("Training and evaluation finished")
    
    # Show sample predictions
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        preds = (torch.sigmoid(outputs) > 0.5).float().cpu()

    classes = ['Banana', 'Blueberry', 'Cherimoya', 'Lemon', 'Lychee', 'Peach', 'Pineapple', 'Raspberry', 'Strawberry', 'Watermelon']
    show_predictions(model, device, dataset2, classes, n=6)

    # Plot Loss
    plt.figure(figsize=(8,5))
    plt.plot(train_losses_list, label='Train Loss')
    plt.plot(test_losses_list, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Label-wise Loss per Epoch')
    plt.legend()
    plt.savefig("Loss.png")
    plt.show()
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(8,5))
    plt.plot(train_accuracies_list, label='Train Accuracy')
    plt.plot(test_accuracies_list, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Label-wise Accuracy per Epoch')
    plt.legend()
    plt.savefig("Accuracy.png")
    plt.show()
    plt.close()

if __name__ == '__main__':
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('CNN Exercise.')
    parser.add_argument('--learning_rate',
                        type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=32,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--batch_size',
                        type=int, default=16,
                        help='Batch size. Must divide evenly into the dataset sizes.')
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help='Directory to put logging.')
    
    FLAGS = None
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)
    
    