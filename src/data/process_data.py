import torch
import csv
from collections import defaultdict
from urllib.error import HTTPError
from time import sleep


TEC_EMO_LABELS_MAPPING = {':: surprise': 's',
                          ':: sadness': 't',
                          ':: joy': 'h',
                          ':: disgust': 'd',
                          ':: fear': 'f',
                          ':: anger': 'a'}


def read_hasoc_data(hs_data_file):
    hs_data = {}
    hs_labels_cg = defaultdict(int)
    hs_labels_fg = defaultdict(int)
    with open(hs_data_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            hs_data[row['_id']] = (row['text'], row['task_1'], row['task_2'])
            hs_labels_cg[row['task_1']] += 1
            hs_labels_fg[row['task_2']] += 1
    return hs_data, hs_labels_cg, hs_labels_fg


def read_hasoc_test_data(hasoc_test_data, hasoc_test_cg_labels, hasoc_test_fg_labels):
    annotations_cg = {}
    with open(hasoc_test_cg_labels) as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            annotations_cg[row['id']] = row['label']
    annotations_fg = {}
    with open(hasoc_test_fg_labels) as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            annotations_fg[row['id']] = row['label']
    instances = {}
    with open(hasoc_test_data) as csvfile:
        csv_reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in csv_reader:
            instances[row['_id']] = (row['text'], annotations_cg[row['_id']], annotations_fg[row['_id']])
    return instances


def read_tec_data(tec_data_file):
    emo_data = {}
    emo_labels = defaultdict(int)
    with open(tec_data_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            try:
                emo_label = TEC_EMO_LABELS_MAPPING[row[2]]
                text = row[1]
            except KeyError:
                # in four cases: broken csv format where text is split up
                try:
                    emo_label = TEC_EMO_LABELS_MAPPING[row[3]]
                    text = row[1] + ',' + row[2]
                except KeyError:
                    # one case where text is split into three columns
                    emo_label = TEC_EMO_LABELS_MAPPING[row[4]]
                    text = row[1] + ',' + row[2] + ',' + row[3]
            emo_labels[emo_label] += 1
            emo_data[row[0]] = (text, emo_label)
    return emo_data, emo_labels


def read_hs_emo_data(hs_emo_data_file, hs_data, number_of_annotated_instances=1000):
    hs_emo_data = {}
    emo_n_labels = defaultdict(int)
    hs_emo_labels_cg = defaultdict(int)
    hs_emo_labels_fg = defaultdict(int)
    emo_hs_labels_cg = defaultdict(lambda: defaultdict(int))
    emo_hs_labels_fg = defaultdict(lambda: defaultdict(int))
    with open(hs_emo_data_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            if len(hs_emo_data) == number_of_annotated_instances:
                break
            emo_label = row['emotion'].strip()
            emo_n_labels[emo_label] += 1

            hs_label_cg = hs_data[row['_id']][1]
            hs_emo_labels_cg[hs_label_cg] += 1
            emo_hs_labels_cg[hs_label_cg][emo_label] += 1

            hs_label_fg = hs_data[row['_id']][2]
            hs_emo_labels_fg[hs_label_fg] += 1
            emo_hs_labels_fg[hs_label_fg][emo_label] += 1

            hs_emo_data[row['_id']] = (row['text'], emo_label, hs_label_cg, hs_label_fg)
    return hs_emo_data, emo_n_labels, hs_emo_labels_cg, hs_emo_labels_fg, emo_hs_labels_cg, emo_hs_labels_fg


def determine_text_len(data, args, percentile=.99):
    texts = [instance[0] for instance in data.values()]
    tokenizer = ModelTokenizer(args.model_name, 0)
    tokenized_text_lens = [len(tokenizer.tokenize_variable_length(text)) for text in texts]
    print(sorted(tokenized_text_lens)[round(len(tokenized_text_lens)*percentile)])


class ModelTokenizer(object):

    def __init__(self, model_name, max_len):
        super(ModelTokenizer, self).__init__()
        self.max_len = max_len
        tokenizer_loaded = False
        while not tokenizer_loaded:
            try:
                self._tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', model_name)
                tokenizer_loaded = True
            except (HTTPError, ValueError):
                print('HTTPError when loading tokenizer, sleeping for a minute and trying again')
                sleep(60)

    def tokenize_text(self, text):
        return self._tokenizer(text, padding='max_length', max_length=self.max_len, truncation=True,
                               return_tensors="pt")

    def tokenize_variable_length(self, text):
        return self._tokenizer.tokenize(text)
