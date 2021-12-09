from enums import STRATEGY
from active_utils import *
from utils import *
from pnetworks import *
from utils import *
from configs import *
from sklearn.model_selection import train_test_split

def start_active_learning(train, dev, test, model_config):
    ### собрать Init_data
    print("1")
    dataPool = DataPool(train['texts'], train['labels'], init_num=0)
    budget_init = model_config.init_budget
    sum_price_init, price_init = 0, 0
    unselected_texts, unselected_labels = dataPool.get_unselected()

    while budget_init > 10 and len(unselected_texts) > 1:
        unselected_texts, unselected_labels = dataPool.get_unselected()
        tobe_selected_idxs = ActiveStrategy.random_sampling(unselected_texts, model_config.step_budget)
        tobe_selected_idxs, budget_init, price_init = choose_ids_by_price(tobe_selected_idxs, budget_init,
                                                                          unselected_texts)
        sum_price_init += price_init
        dataPool.update_pool()
        dataPool.update(tobe_selected_idxs)
    selected_texts, selected_labels = dataPool.get_selected()
    selected_ids = dataPool.get_selected_id()
    embedings, labels = get_embeding( selected_ids, selected_labels, train['embed'])
    X_train, X_test, y_train, y_test = train_test_split(embedings, labels, test_size=0.2, random_state=42)
    print("2")
    ### предобучить модель
    model = BiLSTM_CRF(model_config)
    # model.load_state_dict(torch.load(save_model_path))
    optimizer = optim.SGD(model.parameters(), model_config.learning_rate, weight_decay=1e-4)

    model, optimizer, loss = train_model(model, optimizer, X_train, y_train,  X_train, y_train, model_config)
    # torch.save(model.state_dict(), save_model_path)


    ### активка цикл

    end_marker, iterations_of_learning, sum_prices, sum_perfect, sum_changed, sum_not_changed, sum_not_perfect, perfect, not_perfect, changed, not_changed, thrown_away, price = False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    print("3")
    while (selected_texts is None) or sum_prices < model_config.budget - 10 and not end_marker:
        # выбрать несколько примеров с помощью активки и разметить их
        iterations_of_learning += 1
        unselected_ids = dataPool.get_unselected_id()
        small_unselected_ids, small_unselected_texts, small_unselected_labels = dataPool.get_unselected_small(model_config.step)
        small_unselected_embedings, _ = get_embeding( np.array(unselected_ids)[small_unselected_ids], small_unselected_labels,
                                                                train['embed'])
        tobe_selected_idxs  = None
        if model_config.select_strategy == STRATEGY.LC:
            scores = model.predict_viterbi_score(small_unselected_embedings)
            tobe_selected_idxs, tobe_selected_scores = ActiveStrategy.lc_sampling(scores, small_unselected_embedings,
                                                                                  model_config.step_budget)
        elif model_config.select_strategy == STRATEGY.MNLP:
            scores = model.predict_mnlp_score(small_unselected_embedings)
            tobe_selected_idxs, tobe_selected_scores, price = ActiveStrategy.mnlp_sampling(scores, small_unselected_embedings,
                                                                                           model_config.step_budget)
        elif model_config.select_strategy == STRATEGY.TTE:
            probs = model.predict_probs(small_unselected_embedings)
            tobe_selected_idxs, tobe_selected_scores = ActiveStrategy.tte_sampling(probs, small_unselected_embedings,
                                                                                   model_config.step_budget)
        elif model_config.select_strategy == STRATEGY.TE:
            probs = model.predict_probs(small_unselected_embedings)
            tobe_selected_idxs, tobe_selected_scores = ActiveStrategy.te_sampling(probs, small_unselected_embedings,
                                                                                  model_config.step_budget)
        elif model_config.select_strategy == STRATEGY.RAND:
            tobe_selected_idxs = ActiveStrategy.random_sampling(small_unselected_embedings,
                                                                model_config.step_budget)

        price = 0
        if model_config.label_strategy == STRATEGY.LAZY: #разметка проверяется оракулом, испольщуем PREDICT, а не GOLD
            scores, predicted_labels = predict_precision_span(model, model_config, small_unselected_embedings, small_unselected_labels)
            tobe_selected_idxs, tobe_selected_scores, thrown_away, perfect, not_perfect, price = ActiveStrategy.sampling_precision(tobe_selected_idxs=tobe_selected_idxs, texts=small_unselected_embedings, scores=scores, threshold=model_config.threshold, step=min(model_config.step, model_config.budget - sum_prices))
            changed, not_changed = dataPool.update_labels(tobe_selected_idxs, small_unselected_ids, predicted_labels)
            tobe_selected_idxs = np.array(small_unselected_ids)[tobe_selected_idxs]
            sum_changed += changed
            sum_not_changed += not_changed
        else: #оракул размечает используем GOLD разметку
            tobe_selected_idxs = np.array(small_unselected_ids)
            tobe_selected_idxs_copy = tobe_selected_idxs.copy()
            tobe_selected_idxs = []
            for id in tobe_selected_idxs_copy:
                cost = len(small_unselected_embedings[id])
                if price + cost > model_config.budget - sum_prices:
                    end_marker = True
                    break
                else:
                    tobe_selected_idxs.append(id)
                    price += cost
            tobe_selected_idxs = np.array(small_unselected_ids)[tobe_selected_idxs]

        sum_prices += price
        sum_perfect += perfect
        sum_not_perfect += not_perfect
        dataPool.update_pool()
        dataPool.update(tobe_selected_idxs)
        selected_texts, selected_labels = dataPool.get_selected()
        selected_ids = dataPool.get_selected_id()
        unselected_texts, unselected_labels = dataPool.get_unselected()


        embedings, labels = get_embeding(selected_ids, selected_labels, train['embed'])
        X_train, X_test, y_train, y_test = train_test_split(embedings, labels, test_size=0.2, random_state=42)
        fullcost = compute_price(selected_labels)
        print("4")
        # обучить новую модель
        model = BiLSTM_CRF(model_config)
        # model.load_state_dict(torch.load(save_model_path))
        optimizer = optim.SGD(model.parameters(), model_config.learning_rate, weight_decay=1e-4)
        model, optimizer, loss = train_model(model, optimizer, X_train, y_train, X_test, y_test, model_config)
        # torch.save(model.state_dict(), save_model_path)
        print("5")

    print("6")
    model = BiLSTM_CRF(model_config)
    optimizer = optim.SGD(model.parameters(), model_config.learning_rate, weight_decay=1e-4)

    model, optimizer, loss = train_model(model, optimizer, X_train, y_train, X_test, y_test, model_config)


    tags = []
    for test in X_test:
        precheck_sent = prepare_sequence(test)
        _, history = model(precheck_sent)
        tag = id_to_labels(history, model_config.tag_to_ix)
        tags.append(tag)

    pr, re, f1 = model.f1_score_span(y_test, tags)
    print(pr, re, f1)