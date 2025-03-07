import torch
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

#-----------------------------------------------------------------------------#
# Define dataset
#-----------------------------------------------------------------------------#
class ScalarDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, target_transform=None):

        features = np.loadtxt('linear_advection_features.csv', delimiter=',')
        labels   = np.loadtxt('linear_advection_labels.csv',   delimiter=',')
        
        features = features.astype(np.float32)
        labels   = labels.astype(np.float32)

        self._x = features
        self._y = labels

        print(f"Shape of features: {features.shape}")
        print(f"Shape of labels: {labels.shape}")


        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self._x[:, 0])

    def __getitem__(self, idx):
        x = self._x[idx, :]
        y = self._y[idx]

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


#-----------------------------------------------------------------------------#
# Define model:
#-----------------------------------------------------------------------------#
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._linear1 = torch.nn.Linear(2, 1, bias=False)

    def forward(self, x):
        x1 = self._linear1(x)
        return x1

    def print(self):
        print(self._linear1.weight.data)
        print(self._linear1.bias)



#-----------------------------------------------------------------------------#
# Define training function:
#-----------------------------------------------------------------------------#
def train(dataloader, model, loss_fn, optimizer, scheduler,iter_tol=1e-6):
    size = len(dataloader.dataset)
    model.train()
    loss_values = []
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        
        model.print()
        
        loss_value, current = loss.item(), (batch + 1) * len(X)
        loss_values.append(loss_value)

    scheduler.step()

    return loss_values


#-----------------------------------------------------------------------------#
# Define testing function:
#-----------------------------------------------------------------------------#
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    loss_value = 1.0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        X.values = X.cpu().detach().numpy()
        pred.values = pred.cpu().detach().numpy()

        loss_value, current = loss.item(), (batch + 1) * len(X)

    return loss_value


batch_size = 1
num_epochs = 1

training_data    = ScalarDataset()
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

test_data        = ScalarDataset()
test_dataloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

model = NeuralNetwork().to(device)
loss_fn = torch.nn.MSELoss()
#optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    steps_per_epoch=len(training_data),
    epochs=num_epochs,
    verbose=False,
    final_div_factor=1e4,
    anneal_strategy="cos",
)
#scheduler = []

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


#--------------------------------------------------------------------------------#
# Training loop
#--------------------------------------------------------------------------------#
iter_tol  = 1e-10
loss_prev = 1.0
lossall   = []

for epoch in range(num_epochs):
    
    loss = train(train_dataloader, model, loss_fn, optimizer, scheduler)

    print(f"epoch: {epoch:d}, loss: {loss[-1]:>7e}")

    if (np.abs(loss[-1])<iter_tol):
        break

    loss_prev = loss[-1]
    lossall.extend(loss)

#--------------------------------------------------------------------------------#
# Test model
#--------------------------------------------------------------------------------#
model.eval()
with torch.no_grad():
    test_loss = test(test_dataloader, model, loss_fn)
    print(f"Test Error: \n Accuracy: {test_loss:>7f} \n")





#--------------------------------------------------------------------------------#
# Plot loss
#--------------------------------------------------------------------------------#
plt.semilogy(lossall,'.-')
plt.grid(True)
plt.show()
