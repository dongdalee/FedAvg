# main
WORKER_NUM = 5
TOTAL_ROUND = 2
TRAINING_EPOCH = 1

# setup for worker
MINI_BATCH_SIZE = 64
LEARNING_RATE = 0.0001

# data set labels
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # Mnist

ATTACK_TYPE = "FGSM"

# model weight attack
MALICIOUS_NODE_NUM = 3
GAUSSIAN_MEAN = 0
GAUSSIAN_SIGMA = 2

# FGSM attack
EPSILON = 0.9

# PGD attack
ALPHA = 0.9
STEP = 40

SAVE_SHARD_MODEL_PATH = "./model/"

