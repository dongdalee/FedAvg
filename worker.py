import torch.nn.init
import warnings
import torch.nn.init
from torch import nn
from functools import reduce
import os
import torchattacks

from model import CNN
from dataloader import get_dataloader
from util import gaussian_distribution
import parameter as p
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
warnings.filterwarnings(action='ignore')

# GPU가 없을 경우, CPU를 사용한다.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 랜덤 시드 고정
torch.manual_seed(777)

# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

GAUSSIAN_RANGE = np.arange(-1, 1, 0.00001)

class Worker:
    def __init__(self, _worker_id, current_round):
        self.worker_id = _worker_id
        print("{0} Setup!".format(self.worker_id))

        self.data_loader, self.test_loader = get_dataloader('worker', self.worker_id)

        self.model = CNN().to(device)
        self.trained_model = CNN()
        if current_round-1 > 0:
            self.trained_model.load_state_dict(torch.load("./model/" + str(current_round - 1) + "/aggregation.pt"), strict=False)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=p.LEARNING_RATE)
        self.total_batch = len(self.data_loader)


        if current_round-1 > 0:
            self.model.load_state_dict(torch.load("./model/" + str(current_round - 1) + "/aggregation.pt"), strict=False)
            print("[{0}]: global model inital".format(self.worker_id))


    def loacl_learning(self, training_epochs=p.TRAINING_EPOCH):
        print('Input training epochs: {0}'.format(training_epochs))

        for epoch in range(training_epochs):
            avg_cost = 0

            for X, Y in self.data_loader:  # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블
                X = X.to(device)
                Y = Y.to(device)

                self.optimizer.zero_grad()
                hypothesis = self.model(X)
                cost = self.criterion(hypothesis, Y)
                cost.backward()
                self.optimizer.step()

                avg_cost += cost / self.total_batch
            print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))


    def weight_poison_attack(self):
        self.model.layer1[0].weight.data += noise_constructor(self.model.layer1[0].weight.size())
        self.model.layer1[0].bias.data += noise_constructor(self.model.layer1[0].bias.size())

        self.model.layer2[0].weight.data += noise_constructor(self.model.layer2[0].weight.size())
        self.model.layer2[0].bias.data += noise_constructor(self.model.layer2[0].bias.size())

        self.model.layer3[0].weight.data += noise_constructor(self.model.layer3[0].weight.size())
        self.model.layer3[0].bias.data += noise_constructor(self.model.layer3[0].bias.size())

        self.model.fc1.weight.data += noise_constructor(self.model.fc1.weight.size())
        self.model.fc1.bias.data += noise_constructor(self.model.fc1.bias.size())

        self.model.fc2.weight.data += noise_constructor(self.model.fc2.weight.size())
        self.model.fc2.bias.data += noise_constructor(self.model.fc2.bias.size())


    def FGSM_attack(self, training_epochs=p.TRAINING_EPOCH):
        print('Input training epochs: {0}'.format(training_epochs))

        for epoch in range(training_epochs):
            avg_cost = 0

            fgsm = torchattacks.FGSM(self.trained_model, eps=p.EPSILON)

            for data, target in self.data_loader:
                data, target = data.to(device), target.to(device)
                data = fgsm(data, target)

                self.optimizer.zero_grad()
                hypothesis = self.model(data)
                cost = self.criterion(hypothesis, target)
                cost.backward()
                self.optimizer.step()

                avg_cost += cost / self.total_batch
            print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))


    def PGD_attack(self, training_epochs=p.TRAINING_EPOCH):
        print('Input training epochs: {0}'.format(training_epochs))

        for epoch in range(training_epochs):
            avg_cost = 0

            pgd = torchattacks.PGD(self.model, eps=p.EPSILON, alpha=p.ALPHA, steps=p.STEP)

            for data, target in self.data_loader:
                data, target = data.to(device), target.to(device)
                data = pgd(data, target)

                self.optimizer.zero_grad()
                hypothesis = self.model(data)
                cost = self.criterion(hypothesis, target)
                cost.backward()
                self.optimizer.step()

                avg_cost += cost / self.total_batch
            print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))


    def data_noise_attack(self, training_epochs=p.TRAINING_EPOCH):
        print('Input training epochs: {0}'.format(training_epochs))

        for epoch in range(training_epochs):
            avg_cost = 0

            for data, target in self.data_loader:  # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
                data, target = data.to(device), target.to(device)
                data = add_noise(data, p.NOISE_SIGMA)

                self.optimizer.zero_grad()
                hypothesis = self.model(data)
                cost = self.criterion(hypothesis, target)
                cost.backward()
                self.optimizer.step()

                avg_cost += cost / self.total_batch
            print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

# 가우시안 노이즈 생성
def noise_constructor(dim):
    tensor_length = reduce(lambda x, y: x * y, dim)
    gaussian = gaussian_distribution(GAUSSIAN_RANGE, p.GAUSSIAN_MEAN, p.GAUSSIAN_SIGMA)
    noise_vector = np.random.choice(gaussian, tensor_length, replace=True)

    noise_dim_split = noise_vector.reshape(dim)
    noise_tensor = torch.Tensor(noise_dim_split)

    return noise_tensor

# for data poisoning attack
def add_noise(_data, sigma):
    noise = torch.randn_like(_data) * sigma # 정규분포상의 랜덤값
    contaminate_data = _data + noise

    return contaminate_data




