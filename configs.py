from enums import STRATEGY
import psutil, os

class ModelConfig(object):
    def __init__(self):

        self.embedding_dim = 768
        self.hidden_dim = 256
        self.tag_to_ix = {"<START>": 0, "<STOP>": 1, 'B-LOC': 2, 'I-LOC': 3, 'B-PER': 4, 'I-PER': 5, 'B-ORG': 6, 'I-ORG': 7, 'B-MISC': 8, 'I-MISC': 9, "O": 10}
        self.learning_rate = 0.003
        self.init_budget = 5000
        self.step_budget = 1500
        self.stop_criteria_steps = 14
        self.budget = 15000
        self.select_strategy = STRATEGY.PRECISION
        self.label_strategy = ""
        self.threshold = 0
        self.select_strategy = STRATEGY.RAND
        self.embed_strategy = 'bert'
        self.save_model_path = "saved_models/model.pth"
        self.loginfo = "./logs/loginfo.csv"
        self.p = psutil.Process(os.getpid())

        # self.embed_dim = 300
        # self.dropout = 0.5
        # self.lstm_size = 256
        # self.lstm_layer = 1
        # self.vocab_size = 30000
        # self.use_pretrained = True
        # self.embed_strategy= 'bert'
        # self.num_oov_buckets = 1
        # self.embed_path = ["data/english/embeding/train_vectors","data/english/embeding/test_vectors","data/english/embeding/dev_vectors"]
        # self.vectors = []
        # self.train_sentences = 'data/english/train_vectors_small.txt'
        # self.vocab = './data/chinese/vocab.txt'
        # self.positive_tags = ['B-LOC', 'I-LOC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC']
        # self.tag_to_ix = {"<START>": 0, "<STOP>": 1, 'B-LOC': 2, 'I-LOC': 3, 'B-PER': 4, 'I-PER': 5, 'B-ORG': 6, 'I-ORG': 7, 'B-MISC': 8, 'I-MISC': 9, "O": 10}
        # self.path = 'logs/loginfo.csv'
        # self.positive_ids = None
        # self.tags = None
        # self.buffer_size = 15000
        # self.epochs = 10
        # self.valid_mode = 'partial'
        # #the amount of data to labelling in one step
        # self.steps_budget= 200
        # #Gradually increasing the step size
        # self.delta_step = 5
        # self.threshold = 0.5
        # self.batch_size = 8
        # self.seed = 10
        # self.stop_criteria_quality = 0.8
        # self.learning_rate = 0.0001
        # self.save_checkpoints_steps = 3000
        # self.model_dir = 'results_prtecision/model_chinese'
        # self.model_dir = None


class ActiveConfig(object):
    def __init__(self, pool_size, select_strategy, budget, step, select_epochs, total_epochs):
        self.pool_size = pool_size
        self.total_num = None
        self.select_num = None
        self.budget = budget
        self.step_budget = 200
        self.delta_step = 0
        self.step = step
        self.select_strategy = select_strategy
        self.select_epochs = select_epochs  # the train epochs each time new samples are added
        self.total_epochs = total_epochs  # the train epochs from scratch when finish sampling
        self.pretrain_epochs = 2
        self.step_unselected_size = 10
        self.update()

    def update(self):
        """
        When you reset any parameters, call this method to update the relevant parameters.
        """
        self.total_num = int(self.budget)
        self.select_num = int(self.step)
