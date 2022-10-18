# main
WORKER_NUM = 10
TOTAL_ROUND = 3
TRAINING_EPOCH = 1

# setup for worker
MINI_BATCH_SIZE = 64
LEARNING_RATE = 0.0001

# data set labels
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # Mnist

AGGREGATION = "KRUM"

ATTACK_TYPE = "MODEL_POISONING"

# for trimmed mean percentage
TRIMMED_MEAN_PERCENT = 0.1

# for krum: for clustering
CLUSTER_NUM = 3

# model weight attack
MALICIOUS_NODE_NUM = 0
GAUSSIAN_MEAN = 0
GAUSSIAN_SIGMA = 2

# FGSM attack
EPSILON = 0.8

# PGD attack
ALPHA = 0.9
STEP = 40

# data noise attack
NOISE_SIGMA = 9

SAVE_SHARD_MODEL_PATH = "./model/"

