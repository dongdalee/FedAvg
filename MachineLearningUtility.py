from model import CNN
import torch
from dataloader import get_dataloader
from parameter import labels, TRIMMED_MEAN_PERCENT
from torch import nn
from geom_median.torch import compute_geometric_median
from functools import reduce
from scipy import stats

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

def median_update(*models):
    server_model = CNN().to()

    layer1_weight = []
    layer1_bias = []

    layer2_weight = []
    layer2_bias = []

    layer3_weight = []
    layer3_bias = []

    fc1_weight = []
    fc1_bias = []

    fc2_weight = []
    fc2_bias = []

    for model in models:
        layer1_weight.append(model.layer1[0].weight.data)
        layer1_bias.append(model.layer1[0].bias.data)

        layer2_weight.append(model.layer2[0].weight.data)
        layer2_bias.append(model.layer2[0].bias.data)

        layer3_weight.append(model.layer3[0].weight.data)
        layer3_bias.append(model.layer3[0].bias.data)

        fc1_weight.append(model.fc1.weight.data)
        fc1_bias.append(model.fc1.bias.data)

        fc2_weight.append(model.fc2.weight.data)
        fc2_bias.append(model.fc2.bias.data)

    layer1_weight_median = compute_geometric_median(layer1_weight, None)
    layer1_bias_median = compute_geometric_median(layer1_bias, None)

    layer2_weight_median = compute_geometric_median(layer2_weight, None)
    layer2_bias_median = compute_geometric_median(layer2_bias, None)

    layer3_weight_median = compute_geometric_median(layer3_weight, None)
    layer3_bias_median = compute_geometric_median(layer3_bias, None)

    fc1_weight_median = compute_geometric_median(fc1_weight, None)
    fc1_bias_median = compute_geometric_median(fc1_bias, None)

    fc2_weight_median = compute_geometric_median(fc2_weight, None)
    fc2_bias_median = compute_geometric_median(fc2_bias, None)

    server_model.layer1[0].weight.data = layer1_weight_median.median
    server_model.layer1[0].bias.data = layer1_bias_median.median

    server_model.layer2[0].weight.data = layer2_weight_median.median
    server_model.layer2[0].bias.data = layer2_bias_median.median

    server_model.layer3[0].weight.data = layer3_weight_median.median
    server_model.layer3[0].bias.data = layer3_bias_median.median

    server_model.fc1.weight.data = fc1_weight_median.median
    server_model.fc1.bias.data = fc1_bias_median.median

    server_model.fc2.weight.data = fc2_weight_median.median
    server_model.fc2.bias.data = fc2_bias_median.median

    return server_model

def trimmed_mean_update(*models):
    resize_layer1_weight = []
    resize_layer1_bias = []

    resize_layer2_weight = []
    resize_layer2_bias = []

    resize_layer3_weight = []
    resize_layer3_bias = []

    resize_fc1_weight = []
    resize_fc1_bias = []

    resize_fc2_weight = []
    resize_fc2_bias = []

    for model in models:
        resize_model_layer1_weight = model.layer1[0].weight.data.reshape(reduce(lambda x, y: x * y, model.layer1[0].weight.data.size())).numpy()
        resize_model_layer1_bias = model.layer1[0].bias.data.reshape(reduce(lambda x, y: x * y, model.layer1[0].bias.data.size())).numpy()

        resize_model_layer2_weight = model.layer2[0].weight.data.reshape(reduce(lambda x, y: x * y, model.layer2[0].weight.data.size())).numpy()
        resize_model_layer2_bias = model.layer2[0].bias.data.reshape(reduce(lambda x, y: x * y, model.layer2[0].bias.data.size())).numpy()

        resize_model_layer3_weight = model.layer3[0].weight.data.reshape(reduce(lambda x, y: x * y, model.layer3[0].weight.data.size())).numpy()
        resize_model_layer3_bias = model.layer3[0].bias.data.reshape(reduce(lambda x, y: x * y, model.layer3[0].bias.data.size())).numpy()

        resize_model_fc1_weight = model.fc1.weight.data.reshape(reduce(lambda x, y: x * y, model.fc1.weight.data.size())).numpy()
        resize_model_fc1_bias = model.fc1.bias.data.reshape(reduce(lambda x, y: x * y, model.fc1.bias.data.size())).numpy()

        resize_model_fc2_weight = model.fc2.weight.data.reshape(reduce(lambda x, y: x * y, model.fc2.weight.data.size())).numpy()
        resize_model_fc2_bias = model.fc2.bias.data.reshape(reduce(lambda x, y: x * y, model.fc2.bias.data.size())).numpy()

        resize_layer1_weight.append(resize_model_layer1_weight)
        resize_layer1_bias.append(resize_model_layer1_bias)

        resize_layer2_weight.append(resize_model_layer2_weight)
        resize_layer2_bias.append(resize_model_layer2_bias)

        resize_layer3_weight.append(resize_model_layer3_weight)
        resize_layer3_bias.append(resize_model_layer3_bias)

        resize_fc1_weight.append(resize_model_fc1_weight)
        resize_fc1_bias.append(resize_model_fc1_bias)

        resize_fc2_weight.append(resize_model_fc2_weight)
        resize_fc2_bias.append(resize_model_fc2_bias)


    agg_layer1_weight = torch.Tensor(stats.trim_mean(resize_layer1_weight, TRIMMED_MEAN_PERCENT)).reshape(models[0].layer1[0].weight.data.size())
    agg_layer1_bias = torch.Tensor(stats.trim_mean(resize_layer1_bias, TRIMMED_MEAN_PERCENT)).reshape(models[0].layer1[0].bias.data.size())

    agg_layer2_weight = torch.Tensor(stats.trim_mean(resize_layer2_weight, TRIMMED_MEAN_PERCENT)).reshape(models[0].layer2[0].weight.data.size())
    agg_layer2_bias = torch.Tensor(stats.trim_mean(resize_layer2_bias, TRIMMED_MEAN_PERCENT)).reshape(models[0].layer2[0].bias.data.size())

    agg_layer3_weight = torch.Tensor(stats.trim_mean(resize_layer3_weight, TRIMMED_MEAN_PERCENT)).reshape(models[0].layer3[0].weight.data.size())
    agg_layer3_bias = torch.Tensor(stats.trim_mean(resize_layer3_bias, TRIMMED_MEAN_PERCENT)).reshape(models[0].layer3[0].bias.data.size())

    agg_fc1_weight = torch.Tensor(stats.trim_mean(resize_fc1_weight, TRIMMED_MEAN_PERCENT)).reshape(models[0].fc1.weight.data.size())
    agg_fc1_bias = torch.Tensor(stats.trim_mean(resize_fc1_bias, TRIMMED_MEAN_PERCENT)).reshape(models[0].fc1.bias.data.size())

    agg_fc2_weight = torch.Tensor(stats.trim_mean(resize_fc2_weight, TRIMMED_MEAN_PERCENT)).reshape(models[0].fc2.weight.data.size())
    agg_fc2_bias = torch.Tensor(stats.trim_mean(resize_fc2_bias, TRIMMED_MEAN_PERCENT)).reshape(models[0].fc2.bias.data.size())


    agg_model = CNN().to(device)

    agg_model.layer1[0].weight.data = agg_layer1_weight
    agg_model.layer1[0].bias.data = agg_layer1_bias

    agg_model.layer2[0].weight.data = agg_layer2_weight
    agg_model.layer2[0].bias.data = agg_layer2_bias

    agg_model.layer3[0].weight.data = agg_layer3_weight
    agg_model.layer3[0].bias.data = agg_layer3_bias

    agg_model.fc1.weight.data = agg_fc1_weight
    agg_model.fc1.bias.data = agg_fc1_bias

    agg_model.fc2.weight.data = agg_fc2_weight
    agg_model.fc2.bias.data = agg_fc2_bias

    return agg_model


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