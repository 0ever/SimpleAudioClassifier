import torch
import torch.nn as nn
from torch import device
from tqdm import tqdm


def training(model, train_dl, num_epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(
                                                        len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        for _, data in enumerate(train_dl):
            inputs = data[0]
            labels = data[1]
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        print(f'Epoch: {epoch+1}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')
    print('Finished training')
