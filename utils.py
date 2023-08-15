import numpy as np
from sklearn.metrics import roc_auc_score
from config import config

# metric
def hit_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    return np.sum(y_true)

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def cal_metric_mind(preds, labels):
    all_auc = []
    all_hit_1 = []
    all_hit_5 = []
    all_hit_10 = []
    all_ndcg_1 = []
    all_ndcg_5 = []
    all_ndcg_10 = []
    all_mrr = []

    for imp, imp_preds in preds.items():
        imp_labels = labels[imp]
        if(np.sum(imp_labels)!= 0):
            '''if all labels are 0 or 1, auc can't be calculated'''
            mrr = mrr_score(y_true=imp_labels, y_score=imp_preds)
            hit_1 = hit_score(y_true=imp_labels, y_score=imp_preds, k=1)
            hit_5 = hit_score(y_true=imp_labels, y_score=imp_preds, k=5)
            hit_10 = hit_score(y_true=imp_labels, y_score=imp_preds, k=10)
            ncg_1 = ndcg_score(y_true=imp_labels, y_score=imp_preds, k=1)
            ncg_5 = ndcg_score(y_true=imp_labels, y_score=imp_preds, k=5)
            ncg_10 = ndcg_score(y_true=imp_labels, y_score=imp_preds, k=10)
            auc = roc_auc_score(y_true=imp_labels, y_score=imp_preds,multi_class='ovo')
            all_auc.append(auc)
            all_hit_1.append(hit_1)
            all_hit_5.append(hit_5)
            all_hit_10.append(hit_10)
            all_ndcg_1.append(ncg_1)
            all_ndcg_5.append(ncg_5)
            all_ndcg_10.append(ncg_10)
            all_mrr.append(mrr)
    res = dict()
    res['auc'] = np.mean(all_auc)
    res['hit@1'] = np.mean(all_hit_1)
    res['hit@5'] = np.mean(all_hit_5)
    res['hit@10'] = np.mean(all_hit_10)
    res['ndcg@1'] = np.mean(all_ndcg_1)
    res['ndcg@5'] = np.mean(all_ndcg_5)
    res['ndcg@10'] = np.mean(all_ndcg_10)
    res['mrr'] = np.mean(all_mrr)

    return res
