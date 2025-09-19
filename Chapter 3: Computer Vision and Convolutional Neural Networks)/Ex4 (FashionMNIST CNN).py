import torch
import torchmetrics
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import logging; from pathlib import Path
import sys; from tqdm.auto import tqdm

SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = .1
N_CLASSES = 10
EPOCHS = 20
SAVE_PATH = Path("/home/frasero/PycharmProjects/Models")
MODEL_NAME = "CNNModelFromCourse(state_dict).pth"
FULL_PATH = SAVE_PATH / MODEL_NAME

# CNN also known ConvNet
# CNN's are known for their capabilities to find patterns in visual data

logging.captureWarnings(True)
Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/CNN_course_logs.log", "w")
    ],
    force=True
)
logger = logging.getLogger(__name__)
sys.excepthook = lambda t, v, tb: logger.error("Uncaught exception", exc_info=(t, v, tb))

print(Path("logs/CNN_course_logs.log").resolve().exists())

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
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.cnn_layers1(x)
        # logger.info(f"Shape after 1st CNN layer: {x.shape}")
        x = self.cnn_layers2(x)
        # logger.info(f"Shape after 2nd CNN layer: {x.shape}")
        x = self.classifier_layer(x)
        # logger.info(f"Shape after classifier layer: {x.shape}")
        return x

data = Data()
image, label = data.train_data[0]
class_names = data.train_data.classes
logger.info(f"Shape of images: {image.shape}\nClass names: {class_names}")
model = Model(input_shape=image.shape[0], output_shape=len(class_names), hidden_units=10).to(device)

# random_images = torch.rand(size=(32,3,64,64))
# test_image = random_images[0]
#
# random_image_tensor = torch.randn(size=(1,28,28)).to(device)
# print(random_image_tensor.shape)
# model(random_image_tensor.unsqueeze(0))

# # =================================================
# print(f"Image batch size: {random_images.shape}")
# print(f"Single image shape: {test_image.shape}")
# print(f"Test image:\n{test_image}")
#
# cnn = nn.Conv2d(in_channels=test_image.shape[0],
#                 out_channels=10,
#                 kernel_size=3,
#                 stride=1,
#                 padding=0)
# cnn_output = cnn(test_image)
# print(cnn_output.shape)
# print(cnn_output)
#
# max_pool = nn.MaxPool2d(kernel_size=2)
# max_pool_output = max_pool(cnn_output)
# print(max_pool_output.shape)
# print(max_pool_output)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=model.parameters(), lr=LEARNING_RATE)
accuracy = torchmetrics.Accuracy("multiclass", num_classes=N_CLASSES).to(device)

def train(epochs:int, model:nn.Module, train_loader:DataLoader):
    for epoch in tqdm(range(epochs)):
        # train mode
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # forward pass
            train_logits = model(x_batch)
            train_predictions = torch.argmax(train_logits, dim=1)

            # loss/accuracy
            loss = criterion(train_logits, y_batch)
            train_loss += loss.item()
            accuracy.update(train_predictions, y_batch)

            # optimizer zero grad
            optimizer.zero_grad()

            # loss backward
            loss.backward()

            # optimizer step
            optimizer.step()
        train_loss /= len(train_loader)
        train_accuracy = accuracy.compute().item()
        accuracy.reset()

        predictions, test_loss, test_accuracy = evaluate(model, data.test_loader)

        logger.info(
            f"Epoch: {epoch} | "
            f"Train loss: {train_loss:.4f} |Train Accuracy: {train_accuracy * 100:.2f}% | "
            f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy * 100:.2f}%"
        )

def evaluate(model:nn.Module, test_loader:DataLoader):
    all_predictions = []
    model.eval()
    test_loss = 0
    with torch.inference_mode():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            test_logits = model(x_batch)
            test_predictions = torch.argmax(test_logits, dim=1)

            loss = criterion(test_logits, y_batch)
            test_loss += loss.item()
            accuracy.update(test_predictions, y_batch)

            all_predictions.append(test_predictions)
        test_loss /= len(test_loader)
        test_accuracy = accuracy.compute().item()
        accuracy.reset()

    predictions = torch.cat(all_predictions, dim=0)
    return predictions, test_loss, test_accuracy

train(EPOCHS, model, data.train_loader)
predictions, test_loss, test_accuracy = evaluate(model, data.test_loader)

logger.info(f"Final loss: {test_loss:.4f} | Final accuracy: {test_accuracy*100:.2f}%")
print(predictions)

SAVE_PATH.mkdir(parents=True, exist_ok=True)
checkpoint = {
    "model_state": model.state_dict(),
    "class_to_idx": data.class_to_idx
}
torch.save(checkpoint, FULL_PATH)
logging.info(f"Model and metadata saved to {FULL_PATH}")