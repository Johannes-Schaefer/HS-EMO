import numpy as np
import torch
from collections import defaultdict
from src.data import datasets


def get_loss(model, output, labels, criteria, tasks=('hate_speech',)):
    if criteria is None:
        return 0., {}
    task_batch_losses = {}

    if 'hate_speech' in tasks:
        hs_criterion = criteria['hate_speech']
        hs_output = output[0]
        try:
            hs_labels = labels.to(model.device)
        except AttributeError:
            hs_labels = labels[0].to(model.device)
        hs_loss = hs_criterion(torch.squeeze(hs_output), torch.squeeze(hs_labels.float()))
        task_batch_losses['hate_speech'] = hs_loss
    elif 'joint_hate_speech' in tasks:
        hs_criterion = criteria['joint_hate_speech']
        hs_output = output[0]
        hs_labels = labels[0].to(model.device)
        hs_loss = hs_criterion(torch.squeeze(hs_output), torch.squeeze(hs_labels.float()))
        task_batch_losses['hate_speech'] = hs_loss

    if 'emotion' in tasks:
        emotion_criterion = criteria.get('emotion', None)
        emotion_output = output[1]
        try:
            emotion_labels = labels.to(model.device)
        except AttributeError:
            emotion_labels = labels[1].to(model.device)
        emotion_loss = emotion_criterion(torch.squeeze(emotion_output), torch.squeeze(emotion_labels.float()))
        task_batch_losses['emotion'] = emotion_loss
    elif 'joint_emotion' in tasks:
        emotion_criterion = criteria.get('joint_emotion', None)
        emotion_output = output[1]
        emotion_labels = labels[1].to(model.device)
        emotion_loss = emotion_criterion(torch.squeeze(emotion_output), torch.squeeze(emotion_labels.float()))
        task_batch_losses['emotion'] = emotion_loss

    total_batch_loss = sum(task_batch_losses.values())
    return total_batch_loss, task_batch_losses


def add_macro_avg(scores, prefix, class_labels):
    for metric in ('prec', 'rec', 'f'):
        scores[prefix + 'm_avg ' + metric] = \
            sum([scores[label + ' ' + metric] for label in class_labels]) / len(class_labels)


def calculate_prf_binary(scores, indices=(1, 0)):
    # returns: scores, {'pos class prec': 0, 'pos class rec': 1, 'pos class f': 2,
    #                   'neg class prec': 3, 'neg class rec': 4, 'neg class f': 5}
    result_scores = {}
    for index in indices:
        if len(indices) > 1:
            res_class = 'pos class' if index == 1 else 'neg class'
        else:
            res_class = ''
        try:
            result_scores[res_class + ' prec'] = scores[0+4*index]\
                                                 / max(float(scores[0+4*index] + scores[2+4*index]), 1.0)
        except ZeroDivisionError:
            result_scores[res_class + ' prec'] = 0.0
        try:
            result_scores[res_class + ' rec'] = scores[0+4*index]\
                                                 / max(float(scores[0+4*index] + scores[3+4*index]), 1.0)
        except ZeroDivisionError:
            result_scores[res_class + ' rec'] = 0.0
        try:
            result_scores[res_class + ' f'] = \
                2 * result_scores[res_class + ' prec'] * result_scores[res_class + ' rec']\
                / max((result_scores[res_class + ' prec'] + result_scores[res_class + ' rec']), 1.0)
        except ZeroDivisionError:
            result_scores[res_class + ' f'] = 0.0
    if len(indices) > 1:
        for metric in ('prec', 'rec', 'f'):
            result_scores['m_avg ' + metric] = \
                (result_scores['pos class ' + metric] + result_scores['neg class ' + metric])/2
    return {k: v if v != np.nan else 0.0 for k, v in result_scores.items()}


def compute_eval_scores(y_pred, y_true, scores):
    scores.append((y_true * y_pred).sum())
    scores.append(((~ y_true) * (~ y_pred)).sum())
    scores.append(((~ y_true) * y_pred).sum())
    scores.append((y_true * (~ y_pred)).sum())


def eval_binary_pred_values(prediction, labels):
    # returns: p_tp, p_tn, p_fp, p_fn, n_tp, n_tn, n_fp, n_fn
    scores = []
    for label_class in (1, 0):
        y_pred = torch.round(prediction) == label_class
        y_true = labels == label_class
        compute_eval_scores(y_pred, y_true, scores)
    return scores


def eval_multi_class_pred_values(prediction, labels, label_set=(1, 0)):
    scores = []
    for label_class in label_set:
        y_pred = torch.argmax(prediction, dim=1) == label_class
        y_true = torch.argmax(labels, dim=1) == label_class
        compute_eval_scores(y_pred, y_true, scores)
    return scores


def eval_pred(model, pred, labels, tasks=('hate_speech', )):
    result_scores = {}
    output_index = 0
    if 'hate_speech_out' in model.output_size and 'hate_speech' in tasks:
        if len(tasks) > 1:
            hs_labels = labels[0].to(model.device)
        else:
            hs_labels = labels.to(model.device)
        if model.output_size['hate_speech_out'] == 1:
            # binary classification
            hs_output = torch.sigmoid(pred[output_index])
            hs_scores = eval_binary_pred_values(torch.squeeze(hs_output), torch.squeeze(hs_labels))
            p_tp, p_tn, p_fp, p_fn, n_tp, n_tn, n_fp, n_fn = hs_scores
            result_scores['hs acc'] = (p_tp + p_tn) / len(hs_labels)
            # result_scores['hs_fp'] = p_fp
            # result_scores['hs_fn'] = p_fn
            for score_name, score in calculate_prf_binary(hs_scores).items():
                result_scores['hs ' + score_name] = score
        else:
            # multi-class classification
            hs_output = torch.nn.functional.softmax(pred[0], dim=1)
            hs_scores = eval_multi_class_pred_values(hs_output, hs_labels,
                                                     label_set=tuple(datasets.HS_LABELS_FG_MAPPING.values()))
            for label, index in datasets.HS_LABELS_FG_MAPPING.items():
                for score_name, score in calculate_prf_binary(hs_scores[4 * index:4 * index + 4], indices=(0,)).items():
                    result_scores[label + score_name] = score
            add_macro_avg(result_scores, prefix='hs ', class_labels=tuple(datasets.HS_LABELS_FG_MAPPING.keys()))
        output_index += 1

    if 'emotion_out' in model.output_size and 'emotion' in tasks:
        if len(tasks) > 1:
            emotion_labels = labels[1].to(model.device)
        else:
            emotion_labels = labels.to(model.device)
        emotion_output = torch.nn.functional.softmax(pred[1], dim=1)
        emotion_scores = eval_multi_class_pred_values(emotion_output, emotion_labels,
                                                      label_set=tuple(datasets.EMO_LABELS_MAPPING.values()))
        for label, index in datasets.EMO_LABELS_MAPPING.items():
            for score_name, score in calculate_prf_binary(emotion_scores[4 * index:4 * index + 4], indices=(0,)).items():
                result_scores[label + score_name] = score
        add_macro_avg(result_scores, prefix='emo ', class_labels=tuple(datasets.EMO_LABELS_MAPPING.keys()))

    return result_scores


