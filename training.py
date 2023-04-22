from datetime import datetime

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from preprocess import load_data
from model import CNN, MyNet

def main():
    # Instantiate the CNN
    cnn = CNN()
    # cnn = MyNet([3, 4, 6, 3])

    # Define the cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    # Define the Adam optimizer
    optimizer = optim.Adam(cnn.parameters(), lr=0.001, betas=(0.9, 0.999))

    training(cnn, epoch=10, optimizer=optimizer, criterion=criterion)


def create_loader(dataset, batch_size=100, shuffle=False):
    # FIXME num_workers
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def training(model, optimizer, criterion, epoch=12, batch_size=100):
    print(f"Epoch: {epoch}")
    print("Loading data...")
    # Load the data
    train_set, test_set, val_set = load_data()
    # print(train_set.data.shape)
    # print(len(train_set))
    train_loader = create_loader(train_set, batch_size=batch_size)
    test_loader = create_loader(test_set, batch_size=batch_size)
    validation_loader = create_loader(val_set, batch_size=batch_size)
    # Create a TensorBoard writer for logging
    writer = SummaryWriter("runs/cnn" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    print("Start training...")
    for ep in range(epoch):
        epoch_loss = 0.0
        for i, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Zero the gradients
            optimizer.zero_grad()
            # print(inputs.shape) torch.Size([batch_size, 1, 28, 28])
            # Forward pass

            outputs = model(inputs)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            epoch_loss += loss.item()

            # Close the TensorBoard writer
            writer.close()
        epoch_loss /= len(train_loader)

        # evaluate accuracy and loss
        train_acc, train_loss = accuracy_loss(model, train_loader, criterion)
        writer.add_scalar('Accuracy/train', train_acc, ep)
        writer.add_scalar('Loss/train', train_loss, ep)
        val_acc, val_loss = accuracy_loss(model, validation_loader, criterion)
        writer.add_scalar('Accuracy/validation', val_acc, ep)
        writer.add_scalar('Loss/validation', val_loss, ep)
        print(f"Epoch {ep + 1}/{epoch}, train loss: {train_loss:.4f}, train accuracy: {train_acc:.4f}")
        print(f"Epoch {ep + 1}/{epoch}, validation loss: {val_loss:.4f}, validation accuracy: {val_acc:.4f}")

    # evaluate accuracy and loss on test set
    # make it a image FIXME
    test_acc, test_loss = accuracy_loss(model, test_loader, criterion)
    writer.add_scalar('Accuracy/test', test_acc, epoch)
    writer.add_scalar('Loss/test', test_loss, epoch)
    print('Finished Training')
    # FIXME augmentation of the test dataset?
    writer.close()


def accuracy_loss(model, data_loader, criterion):
    """return the accuracy and loss of the model on the data_loader"""
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader, start=0):
            inputs, labels = data
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # Compute loss
            loss = criterion(outputs, labels)
            # Update test loss
            loss += loss.item()
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()

    return 100 * correct / total, loss / len(data_loader)



if __name__ == '__main__':
    main()