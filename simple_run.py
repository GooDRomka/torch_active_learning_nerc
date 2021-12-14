from active_utils import *
from configs import *
from simple_learning import start_simple_learning
import sys

model_config = ModelConfig()
train_file = './data/english/train.txt'
test_file = './data/english/test.txt'
dev_file = './data/english/valid.txt'
train_vectors = "./data/english/embeding/train_vectors.txt"
test_vectors = "./data/english/embeding/test_vectors.txt"
dev_vectors = "./data/english/embeding/dev_vectors.txt"
vocab = './data/english/vocab.txt'

train = load_data(train_file, train_vectors)
dev = load_data(dev_file, dev_vectors)
test = load_data(test_file, test_vectors)

params = [1000,1500,2000,3000,4000,6000,8000,10000,12000, 15000,20000,25000,30000]
number = find_new_number("logs/simple")
model_config.loginfo = "logs/simple/" + number + "_loginfo.csv"
seed = 0

model_config.save_model_path = "saved_models/simple_model" + number + ".pth"

try:
    seed_cmd = sys.argv[1]
except:
    seed_cmd = "0"

for i in range(5):
        for param in params:
            for j in range(2):
                seed += 1
                if seed>=float(seed_cmd):
                    model_config.init_budget = param
                    model_config.seed = seed
                    stat_in_file(model_config.loginfo, ["\n\n"])
                    stat_in_file(model_config.loginfo, ['BEGIN', 'init_budget', model_config.init_budget, 'seed', model_config.seed ])
                    start_simple_learning(train, dev, test, model_config)





