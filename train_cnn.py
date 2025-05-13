from cnn import MyCNN
from tv_split import *

import torch

cnn = MyCNN().to(device)
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.MSELoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.0001, momentum=0.2)

# train_data = train_data.to(device)
# val_data = val_data.to(device)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=True)
valloader = torch.utils.data.DataLoader(val_data, batch_size=2, shuffle=True)

trainloader = trainloader

for epoch in range(60):
    print(epoch)
    total_loss = 0
    for data in trainloader:

        optimizer.zero_grad()

        inputs, labels = data['features'], data['labels']

        outputs = cnn(inputs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()

        total_loss += loss.item()

        # print(loss)

        # if torch.sum(torch.isnan(loss)) > 0:
        #     print("nan found!")

        optimizer.step()

    print(total_loss)


