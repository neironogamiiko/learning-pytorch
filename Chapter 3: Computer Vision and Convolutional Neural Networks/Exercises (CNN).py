import torch; import torchmetrics
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import random
import matplotlib.pyplot as plt
import matplotlib; matplotlib.use("TkAgg")
import logging; import sys
from pathlib import Path

logging.captureWarnings(True)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.INFO)
Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/Exercises(CNN).log", "w")
    ],
    force=True
)
logger = logging.getLogger(__name__)
sys.excepthook = lambda t, v, tb: logger.error("Uncaught exception", exc_info=(t, v, tb))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load the torchvision.datasets.MNIST() train and test datasets.

train_data = datasets.MNIST(
    root="data",
    train=True,
    transform=ToTensor(),
    target_transform=None,
    download=True
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    transform=ToTensor(),
    target_transform=None,
    download=True
)

# <-----------------------
#    HYPER PARAMETERS    #
# <-----------------------

BATCH_SIZE = 32
NUM_CLASSES = len(train_data.classes)
LEARNING_RATE = .01
EPOCHS = 25

image, _ = train_data[0]
print(f"Image shape: {image.shape}")

# 2. Visualize at least 5 different samples of the MNIST training dataset.

samples = random.sample(list(train_data), k=5)
plt.figure(figsize=(12,3))
for i, (image, label) in enumerate(samples):
    plt.subplot(1,5, i+1)
    plt.imshow(image.squeeze(),cmap="gray")
    plt.title(train_data.classes[label], fontsize=10)
    plt.axis(False)
plt.show()

# Turn the MNIST train and test datasets into dataloaders using torch.utils.data.DataLoader, set the batch_size=32.

train_loader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    dataset=test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Recreate model used in this notebook (the same model from the CNN Explainer website, also known as TinyVGG) capable of fitting on the MNIST dataset.

class Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.block_one = nn.Sequential(
            #    BLOCK 1    #
            nn.Conv2d(in_channels=1,
                      out_channels=10,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU()
        )

        self.block_two = nn.Sequential(
            #    BLOCK 2    #
            nn.Conv2d(in_channels=10,
                      out_channels=10,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.block_three = nn.Sequential(
            #    BLOCK 3    #
            nn.Conv2d(in_channels=10,
                      out_channels=10,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU()
        )

        self.block_four = nn.Sequential(
            #    BLOCK 4    #
            nn.Conv2d(in_channels=10,
                      out_channels=10,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifiction = nn.Sequential(
            #    CLASSIFICATION    #
            nn.Flatten(),
            nn.Linear(10 * 7 * 7, num_classes)
        )

    def forward(self, x):
        x = self.block_one(x)
        x = self.block_two(x)
        x = self.block_three(x)
        x = self.block_four(x)
        x = self.classifiction(x)

        return x

model = Model(num_classes=NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

metrics = torchmetrics.MetricCollection({
    "accuracy" : torchmetrics.classification.MulticlassAccuracy(num_classes=NUM_CLASSES),
    "precision" : torchmetrics.classification.MulticlassPrecision(num_classes=NUM_CLASSES, average="macro"),
    "recall" : torchmetrics.classification.MulticlassRecall(num_classes=NUM_CLASSES, average="macro"),
    "F1" : torchmetrics.classification.MulticlassF1Score(num_classes=NUM_CLASSES, average="macro")
}).to(device)

def train(epochs: int,
          device: torch.device,
          model: nn.Module,
          train_loader: DataLoader,
          eval_flag: bool = False) -> None:

    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        metrics.reset()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            logits = model(x_batch)

            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions = torch.argmax(logits, dim=1)
            metrics.update(predictions, y_batch)
            train_loss += loss.item()
        train_metrics = metrics.compute()
        train_loss /= len(train_loader)
        logger.info(f"Epoch: {epoch+1} / {epochs} | Loss: {train_loss:.4f} | "
              f"Accuracy: {train_metrics['accuracy']:.4f} | "
              f"Precision: {train_metrics['precision']:.4f} | "
              f"Recall: {train_metrics['recall']:.4f} | "
              f"F1: {train_metrics['F1']:.4f}")
        if eval_flag:
            test(device, model, test_loader)

def test(device: torch.device,
         model: nn.Module,
         test_loader: DataLoader):

    model.eval()
    test_loss = 0
    with torch.inference_mode():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)

            loss = criterion(logits, y_batch)

            predictions = torch.argmax(logits, dim=1)

            metrics.update(predictions, y_batch)
            test_loss += loss.item()

    test_metrics = metrics.compute()
    test_loss /= len(train_loader)
    logger.info(f"Test Loss: {test_loss:.4f} | "
          f"Test Accuracy: {test_metrics['accuracy']:.4f} | "
          f"Test Precision: {test_metrics['precision']:.4f} | "
          f"Test Recall: {test_metrics['recall']:.4f} | "
          f"Test F1: {test_metrics['F1']:.4f}")

train(epochs=EPOCHS,
      device=device,
      model=model,
      train_loader=train_loader,
      eval_flag=True)

logger.info("\nFinal results for test data:\n")

test(device=device,
     model=model,
     test_loader= test_loader)