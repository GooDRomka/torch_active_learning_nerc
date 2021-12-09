from utils import *
from configs import *
from active_learning import start_active_learning

model_config = ModelConfig()
save_model_path = "saved_models/model.pth"
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

start_active_learning(train, dev, test, model_config)





