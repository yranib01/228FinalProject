import torch
from cnn import BigCNN
from tv_split import train_set, val_set, device

# model     = MyCNN().to(device)
model     = BigCNN().to(device);
# print(model)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # momentum=0.2?

trainloader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True)
valloader = torch.utils.data.DataLoader(val_set, batch_size=64)

for epoch in range(30):
    # print(epoch)
    model.train()
    running_loss, n_train = 0.0, 0
    for batch in trainloader:
        inputs = batch['features'].to(device)
        labels = batch['labels'].to(device).squeeze(-1)   # (B, )
        
        optimizer.zero_grad()
        outputs = model(inputs).squeeze(-1)         # (B, )
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)   # total loss
        n_train += inputs.size(0)                      # total number of samples
    
    train_loss = running_loss / n_train
    # train_loss = running_loss / len(trainloader)

    print(f"Epoch {epoch:02d} | train {train_loss:8.3f}")
    # print(f"Epoch {epoch:02d}  train_loss: {running/len(trainloader):.4f}")