import matplotlib.pyplot as plt
from torch.optim import optimizer
from tqdm.notebook import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torchsummary import summary

# parameters
epoches = 20
BATCH_SIZE = 32
pretrained = False
train_input_path = "./testing_train"
test_input_path = "./test"
valid_input_path = "./valid"
output_path = "./outputs/vgg19_test.pth"
outputs = Path('./outputs')

if not outputs.is_dir():
    outputs.mkdir()

# use GPU

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(torch.version.cuda)
print(torch.cuda.is_available())  # true
print(device)


# save model
def save_model(epochs, model, optimizer, criterion, output_Path='./outputs/model.pth', descr=''):
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
        'descr': descr,
    }, output_Path)


# image augment

train_trans = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    # transforms.RandomRotation(degrees=(30, 70)),

    # transforms.RandomResizedCrop(256),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

valid_trans = train_trans
# read file

train_folder = ImageFolder(train_input_path, transform=train_trans, )
test_folder = ImageFolder(test_input_path, transform=valid_trans, )

# print(len(train_folder.classes))


train_loader = torch.utils.data.DataLoader(train_folder, shuffle=True, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_folder, shuffle=False, batch_size=BATCH_SIZE)

###########################################

vgg = models.vgg19(pretrained=pretrained).to(device)
vgg.classifier[6] = nn.Linear(4096, 100).to(device)

# Freezing all layers except last 15
for param in list(vgg.parameters())[:-15]:
    param.requires_grad = False

loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(vgg.parameters(), lr=1e-4)

print(summary(vgg, (3, 224, 224)))


def mean(l: list):
    return sum(l) / len(l)


def plot_losses_and_acc(train_losses, train_accuracies, valid_losses, valid_accuracies):
    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    axes[0].plot(train_losses, label='train_losses')
    axes[0].plot(valid_losses, label='valid_losses')
    axes[0].set_title('Losses')
    axes[0].legend()

    axes[1].plot(train_accuracies, label='train_losses')
    axes[1].plot(valid_accuracies, label='valid_losses')
    axes[1].set_title('Accuracy')
    axes[1].legend()


def validate(model, valid_data, loss_fn):
    valid_losses, valid_accuracies = [], []
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in tqdm(valid_data, leave=False):
            X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device).long()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            valid_losses.append(loss.item())
            preds = torch.argmax(logits, axis=1)

            valid_accuracies.append(((preds == y_batch).sum() / len(preds)).item())
    return mean(valid_losses), mean(valid_accuracies)


def train(model, train_data, valid_data, loss_fn, opt, epoches=5):
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    for epoch in tqdm(range(epoches)):
        train_loss = []
        train_acc = []
        model.train()
        for X_batch, y_batch in tqdm(train_data, leave=False):
            opt.zero_grad()

            X_batch, y_batch = X_batch.to(device).float(), y_batch.to(device).long()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch, )
            train_loss.append(loss.item())

            pred = torch.argmax(logits, dim=1)
            train_acc.append(((pred == y_batch).sum() / len(pred)).item())
            loss.backward()
            opt.step()

        valid_loss, valid_accuracy = validate(model, valid_data, loss_fn)

        train_accuracies.append(mean(train_acc))
        train_losses.append(mean(train_loss))
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        print(
            f'epoch: {epoch}: train_loss: {mean(train_losses)}, train_acc: {mean(train_acc)}, val_loss: {valid_loss}, val_acc: {valid_accuracy}')
    plot_losses_and_acc(train_losses, train_accuracies, valid_losses, valid_accuracies)
    return model, train_losses, train_accuracies, valid_losses, valid_accuracies


################################################

vgg, train_losses, train_accuracies, valid_losses, valid_accuracies = train(vgg, train_loader, test_loader, loss_fn,
                                                                            opt, epoches=epoches)

save_model(epoches, vgg, opt, loss_fn, output_path, descr='15 unfrozen layers; 1e-4 lr')

valid_folder = ImageFolder(valid_input_path, transform=valid_trans, )
valid_loader = torch.utils.data.DataLoader(valid_folder, shuffle=False, batch_size=BATCH_SIZE)
valid_loss, valid_acc = validate(vgg, valid_loader, loss_fn)
print(valid_loss, valid_acc)
plt.show()
