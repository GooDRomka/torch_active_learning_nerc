
from enums import STRATEGY
from active_utils import *
from utils import *
from networks import *
from utils import *
from configs import *
from sklearn.model_selection import train_test_split
import os, psutil

def start_active_learning(train, dev, test, model_config):
    set_seed(model_config.seed)
    print("\n\n\n\n Strating new exp  \"active\" with params:", 'selecting_strategy', model_config.select_strategy, 'labeling_strategy', model_config.label_strategy, 'budget', model_config.budget, 'init_budget', model_config.init_budget, 'step_budget', model_config.step_budget,
                'threshold', model_config.threshold, 'seed', model_config.seed)

    #### набираем init данные
    dataPool = init_data(DataPool(train['texts'], train['labels'], init_num=0), model_config)
    selected_texts, selected_labels = dataPool.get_selected()
    selected_ids = dataPool.get_selected_id()

    stat_in_file(model_config.loginfo, ["initDist", init_distribution(selected_labels), "initbudget", model_config.init_budget,
                    "initSumPrices", compute_price(selected_labels), "memory", model_config.p.memory_info().rss/1024/1024])

    print("init_distribution", init_distribution(selected_labels),"init_budget", compute_price(selected_labels))


    embedings, labels = get_embeding( selected_ids, selected_labels, train['embed'])
    X_train, X_test, y_train, y_test = train_test_split(embedings, labels, test_size=0.2, random_state=42)

    #### обучаем init модель
    model = BiLSTM_CRF(model_config)
    optimizer = optim.Adam(model.parameters(), model_config.learning_rate)
    model, optimizer, loss, metrics = train_model(model, optimizer, X_train, y_train,  X_test, y_test, dev['embed'], dev['labels'], model_config)
    print("init_model trained, budget", compute_price(selected_labels), "metrics ", metrics)

    stat_in_file(model_config.loginfo,
                     ["TrainInitFinished", "len(selected_texts):", len(selected_texts), "budget:", model_config.budget,"price", "init_budget", compute_price(selected_labels),
                      "devprecision", metrics[0], "devrecall", metrics[1], "devf1", metrics[2], "memory", model_config.p.memory_info().rss/1024/1024])

    ### активка цикл
    end_marker, iterations_of_learning, sum_prices, sum_perfect, sum_changed, sum_not_changed, sum_not_perfect, perfect, not_perfect, changed, not_changed, thrown_away, price = False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    while (selected_texts is None) or sum_prices < model_config.budget - 10 and not end_marker:
        iterations_of_learning += 1

        ### выбрать несколько примеров с помощью активки и разметить их
        dataPool, price, perfect, not_perfect, sum_prices = active_learing_sampling(model, dataPool, model_config, train, sum_prices)
        selected_texts, selected_labels = dataPool.get_selected()
        selected_ids = dataPool.get_selected_id()
        embedings, labels = get_embeding(selected_ids, selected_labels, train['embed'])
        X_train, X_test, y_train, y_test = train_test_split(embedings, labels, test_size=0.2, random_state=42)
        fullcost = compute_price(selected_labels)

        #### обучить новую модель
        model = BiLSTM_CRF(model_config)
        optimizer = optim.Adam(model.parameters(), model_config.learning_rate)
        model, optimizer, loss, dev_metrics = train_model(model, optimizer, X_train, y_train, X_test, y_test, dev['embed'], dev['labels'],  model_config)

        #### сохранить результаты
        print("memory after training", model_config.p.memory_info().rss/1024/1024)
        print("iter ", iterations_of_learning, "finished, metrics edv", metrics)
        stat_in_file(model_config.loginfo,
                 ["SelectIterFinished", iterations_of_learning, "len(selected_texts):", len(selected_texts), "price", compute_price(selected_labels),
                  "iter_spent_budget:", price, "not_porfect:", not_perfect, "thrown_away:", thrown_away, "perfect:", perfect, "total_spent_budget:", sum_prices,
                  "devprecision", dev_metrics[0], "devrecall", dev_metrics[1], "devf1", dev_metrics[2], "memory", model_config.p.memory_info().rss/1024/1024])

    tags_test= get_tags(model, test['embed'], model_config)
    test_pr, test_re, test_f1 = model.f1_score_span(test['labels'], tags_test)

    stat_in_file(model_config.loginfo,
                 ["result", "len(selected_texts):", len(selected_texts), "budget:", model_config.budget, "Init_budget:", model_config.init_budget,
                  "testprecision", test_pr, "testrecall", test_re, "testf1", test_f1, "devprecision", dev_metrics[0], "devrecall", dev_metrics[1], "devf1", dev_metrics[2]])

    print("result", "len(selected_texts):", len(selected_texts), "budget:", model_config.budget, "Init_budget:", model_config.init_budget,
                  "testprecision", test_pr, "testrecall", test_re, "testf1", test_f1, "devprecision", dev_metrics[0], "devrecall", dev_metrics[1], "devf1", dev_metrics[2])
