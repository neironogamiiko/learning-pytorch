import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import logging; from pathlib import Path

SEED = 42
BATCH_SIZE = 32

# CNN also known ConvNet
# CNN's are known for their capabilities to find patterns in visual data

logging.captureWarnings(True)
Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/CNN_high_lvl_overview_log", "w")
    ]
)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(SEED); torch.cuda.manual_seed(SEED)
logger.info(f"Current device: {device}")

class Data:
    def __init__(self):
        self.train_data = datasets.FashionMNIST(
            root="data",  # where to download data to?
            train=True,  # do we want the training dataset?
            download=True,  # do we want to download yes/no?
            transform=ToTensor(),  # how do we want to transform the data
            target_transform=None  # how do we want to transform the labels/targets?
        )

        self.test_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
            target_transform=None
        )

        self.train_loader = DataLoader(
            dataset=self.train_data,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        self.test_loader = DataLoader(
            dataset=self.test_data,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        self.class_to_idx = self.train_data.class_to_idx

class Model(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_units):
        super().__init__()
        self.cnn_layers1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.cnn_layers2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*0,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.cnn_layers1(x)
        logger.info(f"Shape after 1st CNN layer: {x.shape}")
        x = self.cnn_layers2(x)
        logger.info(f"Shape after 2nd CNN layer: {x.shape}")
        x = self.classifier_layer(x)
        return x

data = Data()
image, label = data.train_data[0]
class_names = data.train_data.classes
logger.info(f"Shape of images: {image.shape}\nClass names: {class_names}")
model = Model(input_shape=image.shape[0], output_shape=len(class_names), hidden_units=len(class_names)).to(device)