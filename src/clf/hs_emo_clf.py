import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig
from torch.autograd import Function
from tqdm import tqdm
from src.data import datasets
from src.data import process_data
from src.clf import eval
from sklearn.utils import class_weight
from collections import defaultdict
from urllib.error import HTTPError
from time import sleep

SEED = 844


class HsEmoPredictorBERT(nn.Module):

    def __init__(self, input_size, hidden_size, hate_speech_out=1, dropout=0., bert_model_name='bert-base-uncased'):
        super(HsEmoPredictorBERT, self).__init__()

        self.input_size = input_size
        self.output_size = {'hate_speech_out': hate_speech_out, 'emotion_out': 6}

        self.bert_model_name = bert_model_name
        bert_model_loaded = False
        while not bert_model_loaded:
            try:
                self.bert = transformers.AutoModel.from_pretrained(bert_model_name)
                bert_model_loaded = True
            except (HTTPError, ValueError):
                print('HTTPError when loading bert model, sleeping for 10 minutes and trying again')
                sleep(60*10)

        self.dropout_layer = nn.Dropout(p=dropout)
        self.hs_linear = nn.Linear(hidden_size, self.output_size['hate_speech_out'])
        self.emotion_linear = nn.Linear(hidden_size, self.output_size['emotion_out'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        _, encoding = self.bert(input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                return_dict=False)
        encoding = self.dropout_layer(encoding)
        output_layers = [self.hs_linear(encoding), self.emotion_linear(encoding)]
        return output_layers


def apply_model(texts, model, device):
    mask = texts['attention_mask'].to(device)
    input_id = texts['input_ids'].squeeze(1).to(device)
    output = model(input_id, mask)
    return output


def model_criteria(class_weights):
    criteria = {}
    for task_name, task_class_weights in class_weights.items():
        if task_name.endswith('hate_speech_cg'):
            criteria[task_name.replace('_cg', '')] = nn.BCEWithLogitsLoss(pos_weight=task_class_weights)
        else:
            criteria[task_name.replace('_fg', '')] = nn.CrossEntropyLoss(weight=task_class_weights)
    return criteria


def train(model, train_data, val_data, learning_rate, epochs, batch_size, logger,
          early_stopping=True, more_train_data=None, third_training_data=None, all_tasks=(('hate_speech',), (), ())):
    model_path = None
    class_weights = {}
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    class_weights.update(train_data.class_weights)
    more_train_dataloader = None
    if more_train_data is not None:
        more_train_dataloader = torch.utils.data.DataLoader(more_train_data, batch_size=batch_size, shuffle=False)
        class_weights.update(more_train_data.class_weights)
    third_train_dataloader = None
    if third_training_data is not None:
        third_train_dataloader = torch.utils.data.DataLoader(third_training_data, batch_size=batch_size, shuffle=False)
        class_weights.update(third_training_data.class_weights)

    logger.write('Class weights: ' + str(class_weights), print_text=True)
    device = model.device
    criteria = model_criteria(class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if torch.cuda.is_available():
        model = model.cuda()
        for task_name, criterion in criteria.items():
            criteria[task_name] = criterion.cuda()

    best_val_score = float('-inf')
    epochs_waited = 0
    best_epoch = 0
    for epoch_num in range(epochs):
        more_train_dataloader_iterator = iter(more_train_dataloader) if more_train_dataloader is not None else None
        third_train_dataloader_iterator = iter(third_train_dataloader) if third_train_dataloader is not None else None
        model.train()

        total_pred_train = []
        total_label_train = []
        total_loss_train = 0.
        total_task_losses_train = defaultdict(float)

        more_total_pred_train = []
        more_total_label_train = []
        more_total_loss_train = 0.
        more_total_task_losses_train = defaultdict(float)

        third_total_pred_train = []
        third_total_label_train = []
        third_total_loss_train = 0.
        third_total_task_losses_train = defaultdict(float)

        for train_input, train_label in train_dataloader:
            output = apply_model(train_input, model, device)
            total_pred_train.append(output)
            total_label_train.append(train_label)
            batch_loss, batch_task_losses = eval.get_loss(model, output, train_label, criteria, tasks=all_tasks[0])
            total_loss_train += batch_loss.item()
            for task_name, batch_task_loss in batch_task_losses.items():
                total_task_losses_train[task_name] += batch_task_loss.item()

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

            if more_train_dataloader is not None:
                more_train_input, more_train_label = next(more_train_dataloader_iterator)
                # NOTE: if more_train_data would be smaller than train_data, a check for StopIteration has to be added!
                more_output = apply_model(more_train_input, model, device)
                more_total_pred_train.append(more_output)
                more_total_label_train.append(more_train_label)
                more_batch_loss, more_batch_task_losses = eval.get_loss(model, more_output, more_train_label, criteria,
                                                                        tasks=all_tasks[1])
                more_total_loss_train += more_batch_loss.item()
                for task_name, batch_task_loss in more_batch_task_losses.items():
                    more_total_task_losses_train[task_name] += batch_task_loss.item()

                model.zero_grad()
                more_batch_loss.backward()
                optimizer.step()

            if third_train_dataloader is not None:
                try:
                    third_train_input, third_train_label = next(third_train_dataloader_iterator)
                except StopIteration:
                    third_train_dataloader_iterator = iter(third_train_dataloader)
                    third_train_input, third_train_label = next(third_train_dataloader_iterator)
                third_output = apply_model(third_train_input, model, device)
                third_total_pred_train.append(third_output)
                third_total_label_train.append(third_train_label)
                third_batch_loss, third_batch_task_losses = eval.get_loss(model, third_output, third_train_label,
                                                                          criteria, tasks=all_tasks[2])
                third_total_loss_train += third_batch_loss.item()
                for task_name, batch_task_loss in third_batch_task_losses.items():
                    third_total_task_losses_train[task_name] += batch_task_loss.item()

                model.zero_grad()
                third_batch_loss.backward()
                optimizer.step()

        total_pred_train_lists = []
        for index in range(len(model.output_size)):
            total_pred_train_lists.append(torch.cat([batch_pred[index] for batch_pred in total_pred_train], dim=0))
        total_label_train = torch.cat(total_label_train, dim=0)
        train_result_scores = eval.eval_pred(model, total_pred_train_lists, total_label_train, tasks=all_tasks[0])

        out = f'Epochs: {epoch_num + 1} | Train Loss: {batch_size * total_loss_train / len(train_data.texts): .5f} '
        for task_name, task_batch_losses_sum in total_task_losses_train.items():
            out += f'| Train {task_name} loss: {batch_size * task_batch_losses_sum / len(train_data.texts): .5f} '
        for score_name, score in train_result_scores.items():
            out += f'| Train {score_name}: {score: .3f} '
        if more_train_dataloader is not None:
            more_total_pred_train_lists = []
            for index in range(len(model.output_size)):
                more_total_pred_train_lists.append(
                    torch.cat([batch_pred[index] for batch_pred in more_total_pred_train], dim=0))
            more_total_label_train = torch.cat(more_total_label_train, dim=0)
            more_train_result_scores = eval.eval_pred(model, more_total_pred_train_lists, more_total_label_train,
                                                      tasks=all_tasks[1])
            out += f'| More Train Loss: {batch_size * more_total_loss_train / len(more_train_data.texts): .5f} '
            for task_name, task_batch_losses_sum in more_total_task_losses_train.items():
                out += f'| More Train {task_name} loss:' \
                       f' {batch_size * task_batch_losses_sum / len(more_train_data.texts): .5f} '
            for score_name, score in more_train_result_scores.items():
                out += f'| More Train {score_name}: {score: .3f} '

        model.eval()

        total_pred_val = []
        total_label_val = []
        total_loss_val = 0.
        total_task_losses_val = defaultdict(float)
        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                output = apply_model(val_input, model, device)
                batch_loss, batch_task_losses = eval.get_loss(model, output, val_label, criteria)
                total_loss_val += batch_loss.item()
                for task_name, batch_task_loss in batch_task_losses.items():
                    total_task_losses_val[task_name] += batch_task_loss.item()

                total_pred_val.append(output)
                total_label_val.append(val_label)
        total_pred_val_lists = []
        for index in range(len(model.output_size)):
            total_pred_val_lists.append(torch.cat([batch_pred[index] for batch_pred in total_pred_val], dim=0))
        total_label_val = torch.cat(total_label_val, dim=0)
        val_result_scores = eval.eval_pred(model, total_pred_val_lists, total_label_val)

        out += f'| Val loss: {batch_size * total_loss_val / len(val_data.texts): .5f} '
        for task_name, task_batch_losses_sum in total_task_losses_val.items():
            out += f'| Val {task_name}  loss: {batch_size * task_batch_losses_sum / len(val_data.texts): .5f} '
        for score_name, score in val_result_scores.items():
            out += f'| Val {score_name}: {score: .3f} '

        logger.write(out, print_text=True)

        if early_stopping:
            patience = 3
            min_delta = 0.005

            val_score = val_result_scores['hs m_avg f']

            model_path = logger.logfile_path.with_suffix('.pt')

            if val_score < (best_val_score + min_delta):
                logger.write(f'Early stopping check: monitored value did not improve substantially '
                             f'{val_score} < {best_val_score+min_delta}',
                             print_text=True)
                if epochs_waited >= patience:
                    es_string = f'Early stopping (patience={patience} and min_delta={min_delta: .5f}), ' \
                                f'reloaded model from epoch {epoch_num - patience}.'
                    logger.write(es_string, print_text=True)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    break
                elif epoch_num + 1 == epochs:
                    es_string = f'Max epoch reached without improvement in last epoch, ' +\
                                f'reloaded model from epoch {best_epoch}.'
                    logger.write(es_string, print_text=True)
                    model.load_state_dict(torch.load(model_path, map_location=device))
                else:
                    # continue training
                    epochs_waited += 1
            else:
                logger.write(f'Early stopping check: monitored value improved {val_score} > {best_val_score+min_delta}'
                             f', saving model from epoch {epoch_num + 1}', print_text=True)
                epochs_waited = 0
                best_val_score = val_score
                torch.save(model.state_dict(), model_path)
                best_epoch = epoch_num + 1
    return model_path, criteria


def predict(model, test_data, batch_size, logger, criteria=None, write_pred=False, tasks=('hate_speech',)):
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    device = model.device
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    total_pred_test = []
    total_label_test = []
    total_loss_test = 0.
    total_task_losses_test = defaultdict(float)
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            output = apply_model(test_input, model, device)
            if criteria is not None:
                batch_loss, batch_task_losses = eval.get_loss(model, output, test_label, criteria)
                total_loss_test += batch_loss.item()
                for task_name, batch_task_loss in batch_task_losses.items():
                    total_task_losses_test[task_name] += batch_task_loss.item()
            total_pred_test.append(output)
            total_label_test.append(test_label)
    total_pred_test_lists = []
    for index in range(len(model.output_size)):
        total_pred_test_lists.append(torch.cat([batch_pred[index] for batch_pred in total_pred_test], dim=0))
    try:
        total_label_test = torch.cat(total_label_test, dim=0)
    except TypeError:
        total_label_test = (torch.cat([label[0] for label in total_label_test], dim=0),
                            torch.cat([label[1] for label in total_label_test], dim=0))
    test_result_scores = eval.eval_pred(model, total_pred_test_lists, total_label_test, tasks=tasks)

    if write_pred:
        write_pred = write_pred if write_pred else logger.logfile_path.split['.'][0] + '-pred'
        torch.save((model.output_size, total_pred_test_lists, total_label_test, device), write_pred)

    out = f'Test loss: {batch_size * total_loss_test / len(test_data.texts): .5f} '
    for task_name, task_batch_losses_sum in total_task_losses_test.items():
        out += f'| Test {task_name} loss: {batch_size * task_batch_losses_sum / len(test_data.texts): .5f} '
    for score_name, score in test_result_scores.items():
        out += f'| Test {score_name}: {score: .4f} '
    logger.write(out, print_text=True)


def prepare_nn_model(args, hate_speech_out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HsEmoPredictorBERT(args.input_size, args.hidden_size, hate_speech_out=hate_speech_out,
                               dropout=args.dropout, bert_model_name=args.model_name)
    if args.load_model is not None:
        model.load_state_dict(torch.load(args.load_model, map_location=device))
    model.to(device)
    return model

