import torch

def validate(model, val_dl):
    model.eval()
    correct_prediction = 0
    total_prediction = 0

    with torch.no_grad():
        for data in val_dl:
            inputs, labels = data[0], data[1]
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
            outputs = model(inputs)
            _, prediction = torch.max(outputs, 1)
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction/total_prediction
    print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')


def predict(model, sample):
    model.eval()
    with torch.no_grad():
        sample = torch.unsqueeze(sample, 0)
        sample_m, sample_s = sample.mean(), sample.std()
        sample = (sample - sample_m) / sample_s
        output = model(sample)
        prediction = torch.nn.functional.softmax(output, 1)
        conf, classes = torch.max(prediction, dim=1)
        print(f"Confidence: {conf.item() * 100:.2f}%")
        return classes[0].item()
