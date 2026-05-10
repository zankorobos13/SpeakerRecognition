import torch
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss, Conv2d, MaxPool2d, Flatten
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader

def normalize_for_cnn(X):
    """
    X: [N, 80, 361] log-mel spectrograms
    returns: normalized tensor for CNN
    """

    X = X.float()

    # per-sample normalization (CMVN style)
    mean = X.mean(dim=(1, 2), keepdim=True)
    std = X.std(dim=(1, 2), keepdim=True) + 1e-8

    X = (X - mean) / std

    return X

dataset_path = "./data/dataset.pt"

alpha = 0.01
batch_size = 32
epochs = 80
device = "cuda"

data = torch.load(dataset_path)

X = data["X"]
y = data["y"]

indices = torch.randperm(len(y))

X = torch.tensor(X[indices]).to(device)
y = torch.tensor(y[indices]).to(device)

X = normalize_for_cnn(X)

X = X.unsqueeze(1)


X_train, X_val = X[:3800], X[3800:]
y_train, y_val = y[:3800], y[3800:]

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size
)

model = Sequential(
    Conv2d(1, 16, 3, padding="same"),
    ReLU(),
    MaxPool2d(2),
    Conv2d(16, 32, 3, padding="same"),
    ReLU(),
    MaxPool2d(2),
    Conv2d(32, 64, 3, padding="same"),
    ReLU(),
    MaxPool2d(2),
    Flatten(),
    Linear(64*10*45, 128),
    ReLU(),
    Linear(128, 10)
).to(device)

loss_fn = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr = alpha)

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # forward
        y_hat = model(batch_X)

        # loss
        loss = loss_fn(y_hat, batch_y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f}")

# -----------------------------
# VALIDATION
# -----------------------------

model.eval()

correct = 0
total = 0

with torch.no_grad():

    for batch_X, batch_y in val_loader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        y_hat = model(batch_X)

        pred = y_hat.argmax(dim=1)

        correct += (pred == batch_y).sum().item()
        total += len(batch_y)

accuracy = correct / total * 100

print()
print(f"Validation accuracy: {accuracy:.2f}%")