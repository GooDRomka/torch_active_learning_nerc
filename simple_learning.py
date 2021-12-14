
from enums import STRATEGY
from active_utils import *
from utils import *
from networks import *
from utils import *
from configs import *
from sklearn.model_selection import train_test_split
import os, psutil

def start_simple_learning(train, dev, test, model_config):
    set_seed(model_config.seed)
    print("\n\n\n\n Strating new exp \"simple\"with params:", 'init_budget', model_config.init_budget, 'seed', model_config.seed)


    dataPool = init_data(DataPool(train['texts'], train['labels'], init_num=0), model_config)
    selected_texts, selected_labels = dataPool.get_selected()
    selected_ids = dataPool.get_selected_id()

    stat_in_file(model_config.loginfo, ["initDist", init_distribution(selected_labels), "initbudget", model_config.init_budget,
                    "initSumPrices", compute_price(selected_labels), "memory", model_config.p.memory_info().rss/1024/1024])

    print("init_distribution", init_distribution(selected_labels), "sum_prices", compute_price(selected_labels))


    embedings, labels = get_embeding(selected_ids, selected_labels, train['embed'])
    X_train, X_dev, y_train, y_dev = train_test_split(embedings, labels, test_size=0.2, random_state=42)


    model, optimizer, loss, dev_metrics = train_model(X_train, y_train, X_dev, y_dev, dev['embed'], dev['labels'], model_config)


    tags, scores = get_tags(model, test['embed'], model_config)
    test_metrics = model.f1_score_span(test['labels'], tags)

    stat_in_file(model_config.loginfo,
                 ["result", "len(selected_texts):", len(selected_texts), "Init_budget:", model_config.init_budget,
                  "testprecision", test_metrics[0], "testrecall", test_metrics[1], "testf1", test_metrics[2], "devprecision", dev_metrics[0], "devrecall", dev_metrics[1], "devf1", dev_metrics[2]])

    print("result", "len(selected_texts):", len(selected_texts), "Init_budget:", model_config.init_budget,
                  "testprecision", test_metrics[0], "testrecall", test_metrics[1], "testf1", test_metrics[2], "devprecision", dev_metrics[0], "devrecall", dev_metrics[1], "devf1", dev_metrics[2])
