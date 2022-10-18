import random
from worker import Worker
from parameter import WORKER_NUM, TOTAL_ROUND, SAVE_SHARD_MODEL_PATH, MALICIOUS_NODE_NUM, labels, ATTACK_TYPE, AGGREGATION
from model import CNN
import os
import torch
from MachineLearningUtility import test_label_predictions, evaluation, fed_avg, median_update, krum_update,trimmed_mean_update
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.manual_seed_all(777)


worker_id_list = []
each_label_acc_list = []
total_acc_list = []

# variables for creating excel files
index = labels
index = index.extend("Accuracy")
columns = np.arange(1, TOTAL_ROUND+1, 1)
accuracy0 = []
accuracy1 = []
accuracy2 = []
accuracy3 = []
accuracy4 = []
accuracy5 = []
accuracy6 = []
accuracy7 = []
accuracy8 = []
accuracy9 = []

models = []

for worker_index in range(WORKER_NUM):
    worker_id = "worker" + str(worker_index)
    worker_id_list.append(worker_id)

malicious_worker = random.sample(worker_id_list, MALICIOUS_NODE_NUM)
print("Malicious Node: ", malicious_worker)

for round in range(TOTAL_ROUND):
    print("==================== Round{0} ====================".format(round + 1))
    os.mkdir(SAVE_SHARD_MODEL_PATH + str(round + 1))

    # create worker nodes
    workers = []

    for worker_index in range(WORKER_NUM):
        worker_id = "worker" + str(worker_index)
        worker_id_list.append(worker_id)
        worker = Worker(worker_id, round+1)
        workers.append(worker)

    for worker in workers:
        if worker.worker_id in malicious_worker:
            if ATTACK_TYPE == "MODEL_POISONING":
                print("~~~~~~~~~~~~~ {0} Model Poisoning attack ~~~~~~~~~~~~~".format(worker.worker_id))
                worker.weight_poison_attack()
            elif ATTACK_TYPE == "FGSM":
                print("~~~~~~~~~~~~~ {0} FGSM attack ~~~~~~~~~~~~~".format(worker.worker_id))
                worker.FGSM_attack()
            elif ATTACK_TYPE == "PGD":
                print("~~~~~~~~~~~~~ {0} PGD attack ~~~~~~~~~~~~~".format(worker.worker_id))
                worker.PGD_attack()
            elif ATTACK_TYPE == "NOISE_ATTACK":
                print("~~~~~~~~~~~~~ {0} Data Noise attack ~~~~~~~~~~~~~".format(worker.worker_id))
                worker.data_noise_attack()
        else:
            print("------------- {0} training -------------".format(worker.worker_id))
            worker.loacl_learning()

    for worker in workers:
        models.append(worker.model)

    if AGGREGATION == "FEDAVG":
        print("FedAvg update")
        fed_avg_model = fed_avg(*models)
    elif AGGREGATION == "MEDIAN":
        print("Geometric median update")
        fed_avg_model = median_update(*models)
    elif AGGREGATION == "TRIMMED_MEAN":
        print("Trimmed mean update")
        fed_avg_model = trimmed_mean_update(*models)
    elif AGGREGATION == "KRUM":
        print("krum update")
        fed_avg_model = krum_update(*models)
    else:
        print("Wrong AGGREGATION parameter !")

    torch.save(fed_avg_model.state_dict(), "./model/" + str(round + 1) + "/aggregation.pt")

    print(("<" * 15) + " Accuracy " + (">" * 15))
    each_label_acc = test_label_predictions(fed_avg_model)
    accuracy0.append(each_label_acc[0])
    accuracy1.append(each_label_acc[1])
    accuracy2.append(each_label_acc[2])
    accuracy3.append(each_label_acc[3])
    accuracy4.append(each_label_acc[4])
    accuracy5.append(each_label_acc[5])
    accuracy6.append(each_label_acc[6])
    accuracy7.append(each_label_acc[7])
    accuracy8.append(each_label_acc[8])
    accuracy9.append(each_label_acc[9])

    print("-" * 20)
    total_acc = evaluation(fed_avg_model)
    total_acc_list.append(total_acc)

# create Excel file
accuracy_lists = [accuracy0, accuracy1, accuracy2, accuracy3, accuracy4, accuracy5, accuracy6, accuracy7, accuracy8, accuracy9, total_acc_list]

excel_of_accuracy = pd.DataFrame(accuracy_lists, index=index, columns=columns)
file_name = './acc_data/accuracy.xlsx'

excel_of_accuracy.to_excel(file_name)
print('Sales record successfully exported into Excel File')

round = np.arange(0, TOTAL_ROUND+1, 1)
total_acc_list.insert(0, 0)

# plot graph
plt.title("MNIST Federated Learning")
plt.plot(round, total_acc_list, 'o--', label="50%")

plt.xlabel("Global Round")
plt.xticks(range(0, TOTAL_ROUND+1, 5))
plt.ylabel("Accuracy")
plt.yticks([10, 20, 30, 40, 50, 60, 70, 80, 90])
# plt.legend(bbox_to_anchor=(1, 1))
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("graph")
plt.show()








