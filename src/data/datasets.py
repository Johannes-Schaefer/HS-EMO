import numpy as np
import torch
from random import shuffle
from sklearn.utils import class_weight
from collections import OrderedDict


DATA_LABEL_POSITION_MAPPING = OrderedDict((('hate_speech', 0), ('emotion', 1)))
HS_LABELS_CG_MAPPING = OrderedDict((('NOT', 0), ('HOF', 1)))
HS_LABELS_FG_MAPPING = OrderedDict((('NONE', 0), ('PRFN', 1), ('OFFN', 2), ('HATE', 3)))
EMO_LABELS_MAPPING = OrderedDict((('s', 0), ('t', 1), ('h', 2), ('d', 3), ('f', 4), ('a', 5)))


class Dataset(object):

    def __init__(self):
        super(Dataset, self).__init__()
        self.texts = []
        self.labels = []

    def shuffle_data(self):
        data = list(zip(self.texts, self.labels))
        shuffle(data)
        self.texts, self.labels = zip(*data)

    def add_instances(self, data, tokenizer, labels_map, label_pos=1):
        for instance in data.values():
            try:
                self.labels.append(labels_map[instance[label_pos]])
            except KeyError:
                # in cases of illegal labels (e.g. '?' for emo)
                continue
            tokenized_text = tokenizer.tokenize_text(instance[0])
            self.texts.append(tokenized_text)

    def finalize_instances(self, shuffle_data=True, num_label_classes=1):
        if shuffle_data:
            self.shuffle_data()
        if num_label_classes > 1:
            # 1-hot transform labels
            self.labels = np.eye(num_label_classes)[list(self.labels)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch_texts = self.texts[idx]
        batch_labels = self.labels[idx]
        return batch_texts, batch_labels


class HSDataset(Dataset):

    def __init__(self, hs_data, tokenizer, fg_labels=False):
        super(HSDataset, self).__init__()
        self.fg_labels = fg_labels
        self.class_weights = {}
        label_pos = 2 if fg_labels else 1
        hs_labels_map = HS_LABELS_FG_MAPPING if fg_labels else HS_LABELS_CG_MAPPING
        num_label_classes = len(HS_LABELS_FG_MAPPING) if fg_labels else 1
        self.add_instances(hs_data, tokenizer, hs_labels_map, label_pos=label_pos)
        self.calculate_class_weights_hs()
        self.finalize_instances(num_label_classes=num_label_classes)

    def calculate_class_weights_hs(self):
        labels = torch.as_tensor(self.labels)
        hs_class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels.numpy())
        hs_class_weights = torch.tensor(hs_class_weights, dtype=torch.float)
        if self.fg_labels:
            self.class_weights['hate_speech_fg'] = hs_class_weights
        else:
            binary_clf_pos_weight = hs_class_weights[1] / hs_class_weights[0]
            self.class_weights['hate_speech_cg'] = binary_clf_pos_weight


class EmoDataset(Dataset):

    def __init__(self, emo_data, tokenizer):
        super(EmoDataset, self).__init__()
        self.class_weights = {}
        self.add_instances(emo_data, tokenizer, EMO_LABELS_MAPPING, label_pos=1)
        self.calculate_class_weights_emo()
        self.finalize_instances(num_label_classes=len(EMO_LABELS_MAPPING))

    def calculate_class_weights_emo(self):
        labels = torch.as_tensor(self.labels)
        emo_class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels.numpy())
        self.class_weights['emotion'] = torch.tensor(emo_class_weights, dtype=torch.float)


class HSEmoDataset(Dataset):

    def __init__(self, hs_emo_data, tokenizer, fg_labels=False):
        super(HSEmoDataset, self).__init__()
        self.fg_labels = fg_labels
        self.class_weights = {}
        hs_label_pos = 3 if fg_labels else 2
        hs_labels_map = HS_LABELS_FG_MAPPING if fg_labels else HS_LABELS_CG_MAPPING
        self.add_multi_task_instances(hs_emo_data, tokenizer, hs_labels_map, EMO_LABELS_MAPPING,
                                      hs_label_pos=hs_label_pos)
        self.calculate_class_weights_hs_emo()
        num_hs_classes = len(HS_LABELS_FG_MAPPING) if fg_labels else 1
        self.finalize_instances_multi_task(num_label_classes=(num_hs_classes, len(EMO_LABELS_MAPPING)))

    def finalize_instances_multi_task(self, shuffle_data=True, num_label_classes=(1, 6)):
        if shuffle_data:
            self.shuffle_data()
        # 1-hot transform labels
        hs_labels = np.eye(num_label_classes[0])[list([label[0] for label in self.labels])] if num_label_classes[0] > 1\
            else [label[0] for label in self.labels]
        emo_labels = np.eye(num_label_classes[1])[list([label[1] for label in self.labels])]
        self.labels = list(zip(hs_labels, emo_labels))

    def add_multi_task_instances(self, data, tokenizer, hs_labels_map, emo_labels_map, hs_label_pos=1):
        for instance in data.values():
            try:
                emo_label = emo_labels_map[instance[1]]
            except KeyError:
                # in cases of illegal labels (e.g. '?' for emo)
                continue
            tokenized_text = tokenizer.tokenize_text(instance[0])
            self.texts.append(tokenized_text)
            self.labels.append((hs_labels_map[instance[hs_label_pos]], emo_label))

    def calculate_class_weights_hs_emo(self):
        hs_labels = torch.as_tensor([label[0] for label in self.labels])
        hs_class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(hs_labels),
                                                             y=hs_labels.numpy())
        hs_class_weights = torch.tensor(hs_class_weights, dtype=torch.float)
        if self.fg_labels:
            self.class_weights['joint_hate_speech_fg'] = hs_class_weights
        else:
            binary_clf_pos_weight = hs_class_weights[1] / hs_class_weights[0]
            self.class_weights['joint_hate_speech_cg'] = binary_clf_pos_weight

        emo_labels = torch.as_tensor([label[1] for label in self.labels])
        emo_class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(emo_labels),
                                                              y=emo_labels.numpy())
        self.class_weights['joint_emotion'] = torch.tensor(emo_class_weights, dtype=torch.float)


def split_dev_data(dev_data, val_ratio=0.1, exclude_val_ids=()):
    dev_data_ids = list(dev_data.keys())
    shuffle(dev_data_ids)
    train_data = {}
    val_data = {}
    for instance_id, instance in dev_data.items():
        if (len(val_data) < (val_ratio * len(dev_data_ids))) and instance_id not in exclude_val_ids:
            val_data[instance_id] = instance
        else:
            train_data[instance_id] = instance
    return train_data, val_data

