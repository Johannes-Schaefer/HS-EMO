import sys
FOLDER = '.'
sys.path.append(FOLDER)

import torch
from pathlib import Path
import argparse
from src.data import logging
from src.data import process_data
from src.data import datasets
from src.clf import hs_emo_clf


def parse_args():
    parser = argparse.ArgumentParser(
        description='Experiments with hate speech detection in combination with emotion annotated data.')
    parser.add_argument('-m', '--model_name', dest='model_name', default='bert-base-uncased', help='(BERT) model name')
    parser.add_argument('-inl', '--input_max_len', dest='input_size', type=int, default=103,
                        help='max length of data instances')
    parser.add_argument('-hs', '--hidden_size', dest='hidden_size', type=int, default=768,
                        help='BERT model hidden/output size')
    parser.add_argument('-ne', '--num_epochs', dest='num_epochs', type=int, default=10,
                        help='(maximum) number of training epochs')
    parser.add_argument('-bs', '--batch_size', dest='batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('-do', '--dropout', dest='dropout', type=float, default=0.2, help='dropout for clf layer(s)')
    parser.add_argument('-lr', '--learning_rate', dest='learning_rate', type=float, default=1e-6,
                        help='Adam optimizer learning rate')
    parser.add_argument('-lm', '--load_model', dest='load_model', default=None,
                        help='load trained model from given path (no additional training will be done)')
    return parser.parse_args()


def prepare_datasets(data_folder_path, args):
    save_path = data_folder_path / 'prepared_datasets.pt'

    annot_file = data_folder_path / 'HS-Emo - Emotionsannotation_1k.csv'
    hs_data_file = data_folder_path / 'HS-Emo - en_Hasoc2021_train.csv'
    emo_data_file = data_folder_path / 'HS-Emo - tec_dataset.csv'
    hasoc_test_data_file = data_folder_path / 'en_Hasoc2021_test_task1.csv'
    hasoc_test_data_cg_labels_file = data_folder_path / '1A_English_actual_labels.csv'
    hasoc_test_data_fg_labels_file = data_folder_path / '1B_English_actual_labels.csv'

    hs_data, _, _ = process_data.read_hasoc_data(hs_data_file)
    print(len(hs_data))

    hs_emo_data, _, _, _, _, _ = process_data.read_hs_emo_data(annot_file, hs_data, number_of_annotated_instances=1000)
    print(len(hs_emo_data))

    emo_data, _ = process_data.read_tec_data(emo_data_file)
    print(len(emo_data))

    hs_test_data = process_data.read_hasoc_test_data(
        hasoc_test_data_file, hasoc_test_data_cg_labels_file, hasoc_test_data_fg_labels_file)
    print(len(hs_test_data))

    model_tokenizer = process_data.ModelTokenizer(args.model_name, args.input_size)

    # process_data.determine_text_len(hs_data, args)
    # result: 103 (99th percentile in HASOC train)

    hs_train_data, hs_val_data = datasets.split_dev_data(hs_data, val_ratio=0.1, exclude_val_ids=hs_emo_data.keys())

    hs_cg_train_ds = datasets.HSDataset(hs_train_data, model_tokenizer, fg_labels=False)
    hs_cg_val_ds = datasets.HSDataset(hs_val_data, model_tokenizer, fg_labels=False)
    hs_fg_train_ds = datasets.HSDataset(hs_train_data, model_tokenizer, fg_labels=True)
    hs_fg_val_ds = datasets.HSDataset(hs_val_data, model_tokenizer, fg_labels=True)

    hs_cg_emo_ds = datasets.HSEmoDataset(hs_emo_data, model_tokenizer, fg_labels=False)
    hs_fg_emo_ds = datasets.HSEmoDataset(hs_emo_data, model_tokenizer, fg_labels=True)

    emo_ds = datasets.EmoDataset(emo_data, model_tokenizer)

    hs_cg_test_ds = datasets.HSDataset(hs_test_data, model_tokenizer, fg_labels=False)
    hs_fg_test_ds = datasets.HSDataset(hs_test_data, model_tokenizer, fg_labels=True)

    torch.save((hs_cg_train_ds, hs_cg_val_ds, hs_fg_train_ds, hs_fg_val_ds,
                hs_cg_emo_ds, hs_fg_emo_ds,
                emo_ds,
                hs_cg_test_ds, hs_fg_test_ds),
               save_path)


def prepare_experiment(args, data_folder_path, log_folder_path, hate_speech_out=1):
    exp_run_logger = logging.Logger(log_folder_path)
    exp_run_logger.write(str(vars(args)), print_text=False)
    data = torch.load(data_folder_path / 'prepared_datasets.pt')
    model = hs_emo_clf.prepare_nn_model(args, hate_speech_out=hate_speech_out)
    return exp_run_logger, data, model


def run_experiment_hs(exp_run_logger, args, data, model, hs_labels_fine_grained=False):
    if hs_labels_fine_grained:
        _, _, hs_train_ds, hs_val_ds, _, _, _, _, hs_test_ds = data
    else:
        hs_train_ds, hs_val_ds, _, _, _, _, _, hs_test_ds, _ = data
    model_path, criteria = hs_emo_clf.train(model, hs_train_ds, hs_val_ds, args.learning_rate, args.num_epochs,
                                            args.batch_size, exp_run_logger,
                                            all_tasks=(('hate_speech',), (), ()))
    hs_emo_clf.predict(model, hs_test_ds, args.batch_size, exp_run_logger, criteria=criteria)


def run_experiment_hs_emo_alternating(exp_run_logger, args, data, model, hs_labels_fine_grained=False):
    if hs_labels_fine_grained:
        _, _, hs_train_ds, hs_val_ds, _, hs_emo_ds, emo_ds, _, hs_test_ds = data
    else:
        hs_train_ds, hs_val_ds, _, _, hs_emo_ds, _, emo_ds, hs_test_ds, _ = data
    model_path, criteria = hs_emo_clf.train(model, hs_train_ds, hs_val_ds, args.learning_rate, args.num_epochs,
                                            args.batch_size, exp_run_logger, more_train_data=emo_ds,
                                            all_tasks=(('hate_speech',), ('emotion',), ()))
    hs_emo_clf.predict(model, hs_test_ds, args.batch_size, exp_run_logger, criteria=criteria, tasks=('hate_speech', ))
    hs_emo_clf.predict(model, hs_emo_ds, args.batch_size, exp_run_logger, criteria=criteria,
                       tasks=('hate_speech', 'emotion'))


def run_experiment_hs_emo_alternating_jointly(exp_run_logger, args, data, model, hs_labels_fine_grained=False):
    if hs_labels_fine_grained:
        _, _, hs_train_ds, hs_val_ds, _, hs_emo_ds, emo_ds, _, hs_test_ds = data
    else:
        hs_train_ds, hs_val_ds, _, _, hs_emo_ds, _, emo_ds, hs_test_ds, _ = data
    model_path, criteria = hs_emo_clf.train(
        model, hs_train_ds, hs_val_ds, args.learning_rate, args.num_epochs, args.batch_size, exp_run_logger,
        more_train_data=emo_ds, all_tasks=(('hate_speech',), ('emotion',), ('joint_hate_speech', 'joint_emotion')),
        third_training_data=hs_emo_ds)
    hs_emo_clf.predict(model, hs_test_ds, args.batch_size, exp_run_logger, criteria=criteria, tasks=('hate_speech', ))
    hs_emo_clf.predict(model, hs_emo_ds, args.batch_size, exp_run_logger, criteria=criteria,
                       tasks=('hate_speech', 'emotion'))


def run_experiment_hs_emo_jointly(exp_run_logger, args, data, model, hs_labels_fine_grained=False):
    if hs_labels_fine_grained:
        _, _, hs_train_ds, hs_val_ds, _, hs_emo_ds, _, _, hs_test_ds = data
    else:
        hs_train_ds, hs_val_ds, _, _, hs_emo_ds, _, _, hs_test_ds, _ = data
    model_path, criteria = hs_emo_clf.train(
        model, hs_train_ds, hs_val_ds, args.learning_rate, args.num_epochs, args.batch_size, exp_run_logger,
        all_tasks=(('hate_speech',), (), ('joint_hate_speech', 'joint_emotion')),
        third_training_data=hs_emo_ds)
    hs_emo_clf.predict(model, hs_test_ds, args.batch_size, exp_run_logger, criteria=criteria, tasks=('hate_speech', ))
    hs_emo_clf.predict(model, hs_emo_ds, args.batch_size, exp_run_logger, criteria=criteria,
                       tasks=('hate_speech', 'emotion'))


def run_all_experiments(data_folder_path, log_folder_path, args):
    lrs = (1e-7, 2.5e-7, 5e-7, 7.5e-7, 1e-6, 2.5e-6, 5e-6, 7.5e-6, 1e-5, 2.5e-5)
    for exp_hs_fine_grained in (False, True):
        exp_num_hs_classes = 4 if exp_hs_fine_grained else 1
        for lr in lrs:
            args.learning_rate = lr
            exp_logger, exp_all_data, exp_initial_model = prepare_experiment(args, data_folder_path, log_folder_path,
                                                                             hate_speech_out=exp_num_hs_classes)
            run_experiment_hs(exp_logger, args, exp_all_data, exp_initial_model,
                              hs_labels_fine_grained=exp_hs_fine_grained)
        
        for lr in lrs:
            args.learning_rate = lr
            exp_logger, exp_all_data, exp_initial_model = prepare_experiment(args, data_folder_path, log_folder_path,
                                                                             hate_speech_out=exp_num_hs_classes)
             run_experiment_hs_emo_alternating(exp_logger, args, exp_all_data, exp_initial_model,
                                               hs_labels_fine_grained=exp_hs_fine_grained)
        
        for lr in lrs:
            args.learning_rate = lr
            exp_logger, exp_all_data, exp_initial_model = prepare_experiment(args, data_folder_path, log_folder_path,
                                                                             hate_speech_out=exp_num_hs_classes)
            run_experiment_hs_emo_alternating_jointly(exp_logger, args, exp_all_data, exp_initial_model,
                                                      hs_labels_fine_grained=exp_hs_fine_grained)

        for lr in lrs:
            args.learning_rate = lr
            exp_logger, exp_all_data, exp_initial_model = prepare_experiment(args, data_folder_path, log_folder_path,
                                                                             hate_speech_out=exp_num_hs_classes)
            run_experiment_hs_emo_jointly(exp_logger, args, exp_all_data, exp_initial_model,
                                          hs_labels_fine_grained=exp_hs_fine_grained)


if __name__ == '__main__':
    arguments = parse_args()
    mulo_data_folder_path = Path(FOLDER + '../data/')
    mulo_log_folder_path = Path(FOLDER + '../logs/')

    run_all_experiments(mulo_data_folder_path, mulo_log_folder_path, arguments)

    # NOTE: preparing dataset should only be done once (saves sets in file) and then loaded for all experiments
    # prepare_datasets(mulo_data_folder_path, arguments)

    # hs_fine_grained = True
    # num_hs_classes = 4 if hs_fine_grained else 1
    #
    # arguments.num_epochs = 1

    # logger, all_data, initial_model = prepare_experiment(arguments, mulo_data_folder_path, mulo_log_folder_path,
    #                                                      hate_speech_out=num_hs_classes)
    # run_experiment_hs(logger, arguments, all_data, initial_model, hs_labels_fine_grained=hs_fine_grained)
    # run_experiment_hs_emo_alternating(logger, arguments, all_data, initial_model,
    #                                   hs_labels_fine_grained=hs_fine_grained)
    # run_experiment_hs_emo_alternating_jointly(logger, arguments, all_data, initial_model,
    #                                           hs_labels_fine_grained=hs_fine_grained)

