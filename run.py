from active_utils import *
from configs import *
from active_learning import start_active_learning


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

number = find_new_number("logs/active")
model_config.loginfo = "logs/active/" + number + "_loginfo.csv"

params = [[STRATEGY.RAND, STRATEGY.LAZY, 2000, 20000, 0.5],
          [STRATEGY.RAND, STRATEGY.LAZY, 3000, 20000, 0.5],
          [STRATEGY.RAND, STRATEGY.LAZY, 4000, 20000, 0.5],
          [STRATEGY.RAND, STRATEGY.LAZY, 5000, 20000, 0.5],
          [STRATEGY.RAND, STRATEGY.LAZY, 6000, 20000, 0.5],
          [STRATEGY.RAND, STRATEGY.LAZY, 7000, 20000, 0.5],
          ]

seed = 0
for i in range(5):
        for param in params:
            for j in range(2):
                seed += 1
                model_config.select_strategy, model_config.label_strategy, model_config.init_budget, model_config.budget, model_config.threshold = param
                model_config.seed = seed

                stat_in_file(model_config.loginfo, ["\n\n"])
                stat_in_file(model_config.loginfo, ['BEGIN', 'selecting_strategy', model_config.select_strategy, 'labeling_strategy', model_config.label_strategy, 'budget', model_config.budget, 'init_budget', model_config.init_budget, 'step_budget', model_config.step_budget,
                                'threshold', model_config.threshold,  'seed', model_config.seed ])

                start_active_learning(train, dev, test, model_config)





