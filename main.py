import pandas as pd
import torch
import torchaudio
import pathlib
from predictions import predict
from tqdm import tqdm
from torch.utils.data import random_split
from classification_ds import SoundDS
from classification_model import AudioClassifier
from training import training
from predictions import validate
from predictions import predict

if __name__ == "__main__":
    labels = {
        'happiness': 0,
        'sadness': 1,
        'anger': 2,
        'disgust': 3,
        'fear': 4
    }
    data = []
    paths = pathlib.Path("/Users/ignas/desktop/dl1/content/data/aesdd").rglob('**/*.wav')
    paths_list = list(paths)
    for path in tqdm(paths_list):
        label = str(path).split('/')[-2]
        try:
            s = torchaudio.load(path)
            data.append({
                "relative_path": path,
                "classID": labels[label]
            })
        except Exception as e:
            pass

    df = pd.DataFrame(data)
    df.head()
    dataset = SoundDS(df, "/Users/ignas/desktop/dl1/content/data/aesdd")
    num_items = len(dataset)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(dataset, [num_train, num_val])
    train_dl = torch.utils.data.DataLoader(train_ds, num_workers=2, batch_size=16, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_ds, num_workers=2, batch_size=16, shuffle=False)

    model = AudioClassifier()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    next(model.parameters()).device
    num_epochs = 100

    #training(model, train_dl, num_epochs)
    model.load_state_dict(torch.load('aesdd-cnn.pth'))
    validate(model, val_dl)
    #Testing with one audio file
    audio, res = val_ds[50]
    print("\nGiven input: ", list(labels.keys())[res])
    prediction_idx = predict(model, audio)
    print("Predicted: ", list(labels.keys())[prediction_idx])
    #torch.save(model.state_dict(), 'aesdd-cnn.pth')
