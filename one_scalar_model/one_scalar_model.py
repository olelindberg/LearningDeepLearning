import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)


class ScalarDataset(torch.utils.data.Dataset):
    def __init__(self, num_values, transform=None, target_transform=None):
        self._x = np.zeros((num_values, 1), dtype=np.float32)

        self._x[:, 0] = np.float32(np.random.random([num_values]))
        self._y = self._x

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self._x[:, 0])

    def __getitem__(self, idx):
        x = self._x[idx, 0]
        y = self._y[idx, 0]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# Define model
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = torch.nn.Linear(1, 1, bias=True)

    def forward(self, x):
        x1 = self._linear(x)
        x2 = torch.nn.functional.relu(x1)
        return x2


def train(dataloader, model, loss_fn, optimizer, scheduler):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()

        optimizer.zero_grad()

        loss, current = loss.item(), (batch + 1) * len(X)

        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


batch_size = 1
num_epochs = 10
num_values = 10

training_data = ScalarDataset(num_values)
test_data = ScalarDataset(num_values)
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)


model = NeuralNetwork().to(device)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1.0,
    steps_per_epoch=len(training_data),
    epochs=num_epochs,
    verbose=False,
    final_div_factor=1e4,
    anneal_strategy="cos",
)


for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer, scheduler)
#    test(test_dataloader, model, loss_fn)
print("Done!")
