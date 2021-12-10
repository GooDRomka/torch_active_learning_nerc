import random
import numpy as np
import logging
from pnetworks import *
from utils import *
from configs import *
import os, psutil
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPool(object):
    def __init__(self, texts, labels, init_num):
        self.text_pool = np.array(texts)
        self.label_pool = np.array(labels)
        assert len(texts) == len(labels)
        self.pool_size = len(texts)
        # _l表示已标注数据集,_u表示未标注数据集
        self.selected_texts = None
        self.selected_labels = None
        self.unselected_texts = None
        self.unselected_labels = None
        self.selected_idx = sorted(set(random.sample(list(range(self.pool_size)), init_num)))
        self.unselected_idx = sorted(set(range(self.pool_size)) - set(self.selected_idx))
        self.update_pool()

    def update_pool(self):
        self.selected_texts = self.text_pool[self.selected_idx]
        self.selected_labels = self.label_pool[self.selected_idx]
        self.unselected_texts = self.text_pool[self.unselected_idx]
        self.unselected_labels = self.label_pool[self.unselected_idx]

    def update_idx(self, new_selected_idx):
        new_selected_idx = set(new_selected_idx)
        self.selected_idx = sorted(set(self.selected_idx) | new_selected_idx)
        self.unselected_idx = sorted(set(self.unselected_idx) - new_selected_idx)

    def translate_select_idx(self, source_idx):
        target_idx = [self.unselected_idx[idx] for idx in source_idx]
        return target_idx

    def update(self, unselected_idx):
        unselected_idx = self.translate_select_idx(unselected_idx)
        self.update_idx(unselected_idx)
        self.update_pool()

    def get_unselected_small(self, num):
        num = round(num)
        if num >= len(self.unselected_idx):
            return self.unselected_idx, self.unselected_texts, self.unselected_labels
        idxs = list(range(len(self.unselected_idx)))
        small_unselected_idx = sorted(random.sample(idxs, num))
        small_unselected_labels = self.unselected_labels[small_unselected_idx]
        small_unselected_texts = self.unselected_texts[small_unselected_idx]
        return small_unselected_idx, small_unselected_texts, small_unselected_labels


    def get_selected(self):
        return self.selected_texts, self.selected_labels

    def get_selected_id(self):
        return self.selected_idx

    def get_unselected_id(self):
        return self.unselected_idx

    def get_unselected(self):
        return self.unselected_texts, self.unselected_labels

    def update_labels(self, to_be_selected_ids, tobe_small_selected_idxs, predicted_labels):
        perfect, changed = 0, 0
        price = 0
        ll = []
        for id_to_be in to_be_selected_ids:
            trans_id = np.array(self.unselected_idx)[tobe_small_selected_idxs[id_to_be]]
            ll.append(trans_id)
            price+=len(self.label_pool[trans_id])


            if self.label_pool[trans_id] == predicted_labels[id_to_be]:
                perfect+=1
            else:
                changed+=1
            self.label_pool[trans_id] = predicted_labels[id_to_be]



        return changed, perfect

    def replace_label(self, new_text, new_label):
        for id in range(len(self.text_pool)):
            if self.text_pool[id] == new_text:
                self.label_pool[id] = new_label
                break


    def get_label_pool(self):
        return self.label_pool

    def get_text_pool(self):
        return self.text_pool

class ActiveStrategy(object):
    def __init__(self):
        pass

    def get_strategy(self, name):
        if name == 'lc':
            return self.least_confidence
        elif name == 'mnlp':
            return self.mnlp
        else:
            return self.random_sampling

    @classmethod
    def random_sampling(cls, texts, num):
        idxs = list(range(len(texts)))
        if num > len(texts):
            return idxs
        return random.sample(idxs, num)

    @classmethod
    def lc_sampling(cls, viterbi_scores, texts, select_num):
        """
        Least Confidence
        """
        select_num = select_num if len(texts) >= select_num else len(texts)
        seq_lens = np.array([len(text) for text in texts])
        scores = np.array(viterbi_scores)
        scores = scores/seq_lens
        tobe_selected_idxs = np.argsort(scores)[:select_num]
        tobe_selected_scores = scores[tobe_selected_idxs]
        return tobe_selected_idxs, tobe_selected_scores

    @classmethod
    def mnlp_sampling(cls, mnlp_scores, texts, select_num):

        select_num = select_num if len(texts) >= select_num else len(texts)
        scores = np.array(mnlp_scores)
        tobe_selected_idxs = np.argsort(scores)[:select_num]
        tobe_selected_scores = scores[tobe_selected_idxs]
        price = len(tobe_selected_idxs)
        return tobe_selected_idxs, tobe_selected_scores, price

    @classmethod
    def total_token_entropy(cls, prob):
        epsilon = 1e-9
        prob += epsilon
        tte = np.einsum('ij->', -np.log(prob) * prob)
        return tte

    @classmethod
    def tte_sampling(cls, probs, texts, select_num):
        """
        Total token entropy sampling.
        """
        select_num = select_num if len(texts) >= select_num else len(texts)
        tte_scores = np.array([cls.total_token_entropy(prob[:len(text), :])
                               for prob, text in zip(probs, texts)])
        tobe_selected_idxs = np.argsort(tte_scores)[-select_num:]
        tobe_selected_scores = tte_scores[tobe_selected_idxs]
        return tobe_selected_idxs, tobe_selected_scores

    @classmethod
    def te_sampling(cls, probs, texts, select_num):
        select_num = select_num if len(texts) >= select_num else len(texts)
        te_scores = np.array([cls.total_token_entropy(prob[:len(text), :])/len(text)
                              for prob, text in zip(probs, texts)])
        tobe_selected_idxs = np.argsort(te_scores)[-select_num:]
        tobe_selected_scores = te_scores[tobe_selected_idxs]
        return tobe_selected_idxs, tobe_selected_scores

    # 8) тут мы случайным образом выбираем часть датасета и после делаем ленивую разметку и отбрасываем по threshold данные
    # В одной из статей прочитал сравнение методов и оказалось, что рандом чуть ли не лучший алгоритм выбора данных,
    # там сравнивали всякие метрики уверенности и выбирали самые славбые примеры для разметки
    #  а в итоге рандом работал лучше
    @classmethod
    def random_sampling_precision(cls, scores, num, threshold):
        idxs = list(range(len(scores)))
        bad_examples = []
        if num > len(scores):
            samples = idxs
        else:
            samples = random.sample(idxs, num)
        res = []
        price = len(samples)
        perfect, not_perfect = 0,0
        for id in samples:
            if scores[id] > threshold:
                if scores[id] == 1:
                    perfect += 1
                else:
                    not_perfect += 1
                res.append(id)
            else:
                bad_examples.append(id)

        tobe_selected_idxs = res
        tobe_selected_scores = scores[tobe_selected_idxs]
        return tobe_selected_idxs, tobe_selected_scores, bad_examples, price, perfect, not_perfect

    @classmethod
    def sampling_precision(cls, tobe_selected_idxs, texts, scores, threshold, step):
        thrown_away = 0
        price = 0
        res = []
        scores_res = []
        perfect, not_perfect = 0, 0
        for id in tobe_selected_idxs:
            if len(texts[id]) + price < step:
                price += len(texts[id])
                if scores[id] > threshold:
                    if scores[id] == 1:
                        perfect += 1
                    else:
                        not_perfect += 1
                    res.append(id)
                    scores_res.append(scores[id])
                else:
                    thrown_away += 1
        return res, scores_res, thrown_away, perfect, not_perfect, price




def predict_precision_span(model, model_config, texts, labels):
    tags = []
    scores = []
    for label, text in zip(labels, texts):
        pr = 0
        precheck_sent = prepare_sequence(text)
        _, history = model(precheck_sent)
        tag = id_to_labels(history, model_config.tag_to_ix)
        tags.append(tag)
        if len(text) != len(label) or len(label) != len(tag):
            scores.append(pr)
        else:
            pr, re, f1 = model.f1_score_span([label], [tag])
            scores.append(pr)
    return np.array(scores),  np.array(tags)

def key_by_value(d,value):
    for key, val in d.items():
        if val == value:
            return key

def id_to_labels(seq, tag_to_ix):
    labels = []
    for id in seq:
        labels.append(key_by_value(tag_to_ix, id))
    return labels


def train_model(model, optimizer, X_train, y_train, X_test, y_test, X_dev, y_dev, model_config):
    p = psutil.Process(os.getpid())
    f1s = [-1]
    keep_max, best_epoch,epoch = 0, 0, 0
    print("start training, size train", compute_price(y_train), "size test", compute_price(y_test))
    while keep_max < model_config.stop_criteria_steps:
        # print("memory before epoch",p.memory_info().rss/1024/1024)
        for sentence, tags in zip(X_train,y_train):
            model.zero_grad()

            sentence_in = torch.as_tensor(np.array(sentence))
            targets = torch.tensor([model_config.tag_to_ix[t] for t in tags], dtype=torch.long)

            loss = model.neg_log_likelihood(sentence_in, targets)

            loss.backward()
            optimizer.step()
        epoch+=1

        tags = []
        for test in X_test:
            precheck_sent = prepare_sequence(test)
            _, history = model(precheck_sent)
            tag = id_to_labels(history, model_config.tag_to_ix)
            tags.append(tag)
        pr,re,f1 = model.f1_score_span(y_test, tags)

        print(epoch, pr, re, f1, "memory", p.memory_info().rss/1024/1024)
        # print("memory after epoch",p.memory_info().rss/1024/1024)
        keep_max += 1
        if max(f1s) < f1:
            keep_max = 0
            best_epoch = epoch
        f1s.append(f1)
    tags = []
    for text in X_dev:
        precheck_sent = prepare_sequence(text)
        _, history = model(precheck_sent)
        tag = id_to_labels(history, model_config.tag_to_ix)
        tags.append(tag)
    metrics = model.f1_score_span(y_dev, tags)


    return model, optimizer, loss, metrics

