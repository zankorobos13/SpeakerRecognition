import torch
from torch.nn import Sequential, Linear, ReLU, CrossEntropyLoss, Conv2d, MaxPool2d, Flatten, Dropout
from torch.optim import SGD
from torch.utils.data import TensorDataset, DataLoader

dataset_path = "./data/dataset.pt"

alpha = 0.01
batch_size = 32
epochs = 300
device = "cuda"

data = torch.load(dataset_path)

X = data["X"]
y = data["y"]

indices = torch.randperm(len(y))

X = torch.tensor(X[indices]).to(device)
y = torch.tensor(y[indices]).to(device)

mean = X.mean(dim=2)
std  = X.std(dim=2)

X = torch.cat([mean, std], dim=1)
print(X.shape)
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

    Linear(160, 512),
    ReLU(),
    Dropout(0.2),

    Linear(512, 256),
    ReLU(),
    Dropout(0.2),

    Linear(256, 128),
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
    if epoch == 100:
        alpha = 0.001
    elif epoch == 150:
        alpha = 0.0001
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