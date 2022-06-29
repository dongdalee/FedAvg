from model import CNN
import torch
from dataloader import get_dataloader
from parameter import labels
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def fed_avg(*models):
    fed_avg_model = CNN().to(device)

    fed_avg_model.layer1[0].weight.data.fill_(0.0)
    fed_avg_model.layer1[0].bias.data.fill_(0.0)

    fed_avg_model.layer2[0].weight.data.fill_(0.0)
    fed_avg_model.layer2[0].bias.data.fill_(0.0)

    fed_avg_model.layer3[0].weight.data.fill_(0.0)
    fed_avg_model.layer3[0].bias.data.fill_(0.0)

    fed_avg_model.fc1.weight.data.fill_(0.0)
    fed_avg_model.fc1.bias.data.fill_(0.0)

    fed_avg_model.fc2.weight.data.fill_(0.0)
    fed_avg_model.fc2.bias.data.fill_(0.0)

    for model in models:
        fed_avg_model.layer1[0].weight.data += model.layer1[0].weight.data
        fed_avg_model.layer1[0].bias.data += model.layer1[0].bias.data

        fed_avg_model.layer2[0].weight.data += model.layer2[0].weight.data
        fed_avg_model.layer2[0].bias.data += model.layer2[0].bias.data

        fed_avg_model.layer3[0].weight.data += model.layer3[0].weight.data
        fed_avg_model.layer3[0].bias.data += model.layer3[0].bias.data

        fed_avg_model.fc1.weight.data += model.fc1.weight.data
        fed_avg_model.fc1.bias.data += model.fc1.bias.data

        fed_avg_model.fc2.weight.data += model.fc2.weight.data
        fed_avg_model.fc2.bias.data += model.fc2.bias.data

    fed_avg_model.layer1[0].weight.data = fed_avg_model.layer1[0].weight.data / len(models)
    fed_avg_model.layer1[0].bias.data = fed_avg_model.layer1[0].bias.data / len(models)

    fed_avg_model.layer2[0].weight.data = fed_avg_model.layer2[0].weight.data / len(models)
    fed_avg_model.layer2[0].bias.data = fed_avg_model.layer2[0].bias.data / len(models)

    fed_avg_model.layer3[0].weight.data = fed_avg_model.layer3[0].weight.data / len(models)
    fed_avg_model.layer3[0].bias.data = fed_avg_model.layer3[0].bias.data / len(models)

    fed_avg_model.fc1.weight.data = fed_avg_model.fc1.weight.data / len(models)
    fed_avg_model.fc1.bias.data = fed_avg_model.fc1.bias.data / len(models)

    fed_avg_model.fc2.weight.data = fed_avg_model.fc2.weight.data / len(models)
    fed_avg_model.fc2.bias.data = fed_avg_model.fc2.bias.data / len(models)

    return fed_avg_model


def test_label_predictions(model):
    model.eval()
    actuals = []
    predictions = []

    labels_of_acc = []

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    with torch.no_grad():
        train_data_loader, test_data_loader = get_dataloader(type="all")

        for data, target in test_data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)

            actuals.extend(target.view_as(prediction))
            predictions.extend(prediction)

            _, predicted = torch.max(output, 1)
            c = (predicted == target).squeeze()
            for i in range(4):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        for i in range(10):
            label_accuracy = round(100 * class_correct[i] / class_total[i], 2)
            print('accuracy of {0} : {1:.2f}'.format(labels[i], label_accuracy))
            labels_of_acc.append(label_accuracy)

    return labels_of_acc
    # return [i.item() for i in actuals], [i.item() for i in predictions]


def evaluation(model):
    test_loss = 0
    correct = 0

    train_data_loader, test_data_loader = get_dataloader(type="all")
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_data_loader:
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        print("accuracy: {0}".format(100 * correct / len(test_data_loader.dataset)))
        return 100 * correct / len(test_data_loader.dataset)