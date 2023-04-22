import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import tensor
from PIL import Image
import mnist_reader
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Set the random seed
np.random.seed(7374438)


# Define custom dataset
class MyImageDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        # Implement data retrieval logic
        x = self.data[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.data)


# Load the data and output three Dataloaders
def load_data(path='fashion'):
    # print(Y_label.shape)(10000,)
    # print(X_train.shape) (60000, 784)
    train_data, train_label = mnist_reader.load_mnist(path, kind='train')
    test_data, test_label = mnist_reader.load_mnist(path, kind='t10k')
    print("Preprocessing...")
    train_data = preprocess(train_data)
    train_data, train_label, val, val_label = create_extra_validation(train_data, train_label, 1000)

    # Normalize data to desired range
    train_data = 2 * (train_data - train_data.min()) / (train_data.max() - train_data.min()) - 1
    test_data = 2 * (test_data - test_data.min()) / (test_data.max() - test_data.min()) - 1
    val_data = 2 * (val - val.min()) / (val.max() - val.min()) - 1

    train_data = tensor(train_data, dtype=torch.float32)
    test_data = tensor(test_data, dtype=torch.float32)
    val_data = tensor(val_data, dtype=torch.float32)

    # # Normalize the data
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = MyImageDataset(train_data, train_label)
    test_set = MyImageDataset(test_data, test_label)
    val_set = MyImageDataset(val_data, val_label)

    return train_set, test_set, val_set


def preprocess(image_set):
    # Create an instance of RandomHorizontalFlip
    # FIXME could do vertical flips too?
    # transform = transforms.RandomHorizontalFlip(p=0.5)

    # For each image in the training set and the test set
    # zero-pad 4 pixels on each side of the input images and randomly crop 28x28 as input to
    # the network. This is a form of data augmentation.
    pad = 4
    new_img_set = []

    for img in image_set:
        # Convert the flat array to an image
        img = img.reshape((28, 28))

        # Flip the image
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
        # print(img.shape)
        # plt.imshow(img, cmap='gray')
        # plt.show()

        # Zero-pad the image
        img = np.pad(img, ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))

        # randomly crop
        row_ran = np.random.randint(0, 2 * pad + 1)
        col_ran = np.random.randint(0, 2 * pad + 1)
        img = img[row_ran:row_ran + 28, col_ran:col_ran + 28]
        img = np.expand_dims(img, axis=0)
        # print(img.shape) (1, 28, 28)
        # plt.imshow(img)
        # plt.show()

        # make it flat again FIXME NO RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [50, 784]
        new_img_set.append(img)
    return np.array(new_img_set)


def create_extra_validation(train: tensor, label: tensor, size=1000):
    # split the 1000 data into validation, and they are coming from the training set
    # I am doing with np random with uniform distribution
    # Input is tensor
    # FIXME validation against test
    indices = np.random.choice(train.shape[0], size=size, replace=False)

    val = train[indices]
    # print(val.shape)
    val_label = label[indices]
    # print(train.shape)
    # print(len(train))
    train = np.delete(train, indices, axis=0)
    # print(train.shape)
    # print(len(train))
    train_label = np.delete(label, indices, axis=0)
    return train, train_label, val, val_label


def create_batch(train, label, size=10):
    """not used in this project"""
    # Generate a random permutation of indices
    indices = np.random.permutation(train.shape[0])

    # Use the permutation to shuffle both arrays
    train_shuffled = train[indices]
    label_shuffled = label[indices]

    # Create a batch of images
    interval = train.shape[0] // size
    batches = []
    for i in range(size):
        batch_train = train_shuffled[i * interval:(i + 1) * interval]
        # Create a batch of labels
        batch_label = label_shuffled[i * interval:(i + 1) * interval]
        batches.append((batch_train, batch_label))
    return batches
