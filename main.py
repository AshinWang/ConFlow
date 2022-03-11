import argparse
import os
import torch.utils.data.sampler as sampler
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from models import FlowEncoder
from metrics import ComputeMetrics
from data_read import WideDeepDataset, read_data
from trainer import training

def parse_option():
    parser = argparse.ArgumentParser('argument')
    # dataset
    parser.add_argument('--dataset_name', type=str, default='ids17',
                        choices=['ids12', 'unb15', 'ids17'], help='trainse name')

    parser.add_argument('--dataset_path', type=str, default='datasets/CIC-IDS2017/emb_ids17.csv', 
                        help='dataset path')

    parser.add_argument('--testset_name', type=str, default='ids17',
                        choices=['ids12', 'unb15', 'ids17'], help='testset name')

    parser.add_argument('--testset_path', type=str, default='datasets/CIC-IDS2017/emb_ids17.csv', 
                        help='testset path')                    

    parser.add_argument('--nums_fewshot', type=int, default=100,
                        choices=[10, 100, 1000], help='nums of fewshot dataset')

    parser.add_argument('--type_classes', type=str, default='multiclass',
                        choices=['multiclass', 'binary'], help='types of classification')
    
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='testset size')
    
    parser.add_argument('--valid_size', type=float, default=0.2,
                        help='validset size')

    parser.add_argument('--indipendent_valid', action='store_true', default=True,
                        help='indipendent validset')   
    

    # method
    parser.add_argument('--method', type=str, default='scl+ce',
                        choices=['scl+ce', 'ce'], help='choose loss method')

    # temperature
    parser.add_argument('--temp', type=float, default=0.05,
                        help='temperature for loss function')

    # lambda
    parser.add_argument('--weighted_lambda', type=float, default=0.9,
                        help='weighted CE and SCL')

    # other setting
  
    parser.add_argument('--batch_size', type=int, default=1024*8,
                        help='batch_size')

    parser.add_argument('--num_workers', type=int, default=6,
                        help='num of workers to use')

    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs')

    parser.add_argument('--early_stop', type=int, default=50,
                        help='early_stop')

    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='learning rate')

    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='weight decay')
    
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout')

    # logs
    parser.add_argument('--logs_path', type=str, default='./logs/',
                        help='logs path')
    # checkpoint
    parser.add_argument('--checkpoint_path', type=str, default='./ckpt/',
                        help='checkpoint path')
    parser.add_argument('--best_model', type=str, default='ckpt/ids17/best_acc.pt',
                        help='best checkpoint')

    # metric report 
    parser.add_argument('--reports_path', type=str, default='reports',
                        help='report path')
    
    # train or test mode
    parser.add_argument('--train_test_mode', type=str, default='train_test',
                        choices=['train', 'test', 'train_test'], help='train or test mode')
    
    # few-shot learning
    parser.add_argument('--fewshot_train', action='store_true', default=False,
                        help='few-shot train')
    
    # few-shot learning
    parser.add_argument('--cross_test', action='store_true', default=False,
                        help='cross test')

    opt = parser.parse_args()
    return opt

def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path





def main():
    opt = parse_option()
    mk_dir(opt.logs_path)
    mk_dir(opt.checkpoint_path)
    mk_dir(opt.reports_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    dataset = pd.read_csv(opt.dataset_path)

    y = dataset['label'].astype('int')
    X = dataset.drop(columns=['label'])
    num_classes = int(len(y.unique()))

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        shuffle = True, 
                                                        random_state=2022, 
                                                        stratify=y, 
                                                        test_size=opt.test_size)

    if opt.cross_test:
        testset = pd.read_csv(opt.testset_path)
        y = testset['label'].astype('int')
        X = testset.drop(columns=['label'])

        _, X_test, _, y_test = train_test_split(X, y, 
                                                shuffle = True, 
                                                random_state=2022, 
                                                stratify=y, 
                                                test_size=0.2)

    column_idx, embed_input, continuous_cols, train_dataset, valid_dataset, test_dataset = read_data(X_train, 
                                                                                                     X_test, 
                                                                                                     y_train, 
                                                                                                     y_test, 
                                                                                                     valid_state=opt.indipendent_valid, 
                                                                                                     valid_size=opt.valid_size)
    
    BATCH_SIZE = opt.batch_size

    if opt.fewshot_train:
        opt.type_classes = 'binary'
        
        random_sampler = sampler.RandomSampler(data_source=train_dataset)
        y_0 = []
        y_1 = []
        x_0 = []
        x_1 = []
        k =int(opt.nums_fewshot/2)
        
        for index in random_sampler:
            label = train_dataset[index][1]
            if int(label) == 0 and len(y_0) != k:
                x_0.append(train_dataset[index][0])
                y_0.append(train_dataset[index][1])
            elif int(label) > 0 and len(y_1) != k:
                x_1.append(train_dataset[index][0])
                y_1.append(train_dataset[index][1])
            elif len(y_0) == k and len(y_1) == k:
                break

        x_fewshot = np.array(np.append(x_0, x_1, axis=0))
        y_fewshot = np.array(np.append(y_0, y_1, axis=0))
        fewshot_dataset = WideDeepDataset(x_fewshot, y_fewshot)
        print('Length of fewshot dataset', len(fewshot_dataset))
      
        num_classes = 2

        if opt.nums_fewshot <= 100:
            batch_size = int(opt.nums_fewshot/10)
        elif opt.nums_fewshot == 1000:
            batch_size = 10
        else:
            batch_size = 100
        train_loader = DataLoader(dataset=fewshot_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = train_loader
        
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    if opt.cross_test:
        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    else:
        test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Length of Train Loader: ", len(train_loader))
    print("Length of Valid Loader:", len(valid_loader), "\n")
    
    model = FlowEncoder(column_idx, embed_input, continuous_cols, dropout=opt.dropout, num_classes=num_classes)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(device)
    print("Total trainable parameters = ", params)


    if opt.train_test_mode in ['train_test', 'train']:
        model = training(model, 
                        train_loader, 
                        valid_loader,
                        loss_fn=opt.method,
                        cls_mode=opt.type_classes,
                        epochs=opt.epochs, 
                        learning_rate=opt.learning_rate,
                        early_stop=opt.early_stop,
                        temperature=opt.temp,
                        alpha=opt.weighted_lambda,
                        dataset=opt.dataset_name,
                        weight_decay=opt.weight_decay,
                        )
    if opt.train_test_mode in ['train_test', 'test']:
        def load_weight(ckpt_path):
            model = FlowEncoder(column_idx, embed_input, continuous_cols, dropout=opt.dropout, num_classes=num_classes)
            model.to(device)
            weight = torch.load(ckpt_path, map_location=None)
            model.load_state_dict(weight)  
            return model

        model = load_weight(opt.best_model)
        metrics_report = ComputeMetrics(opt.type_classes, model, test_loader, device, average='macro', dataset=opt.dataset_name)
        cr = metrics_report.cr()
        cm = metrics_report.cm()

        print('Confusion Matrix')
        print(cm)
        print('Classification Report')
        print(cr)
        plt.subplots(figsize=(12, 10.5))
        sns.heatmap(cm, annot=True, fmt='0.6g',  linewidths=1 ,cmap='gist_earth_r', linecolor='#666666',
                        xticklabels=metrics_report.target_names, yticklabels=metrics_report.target_names)
        plt.savefig("{}/{}.png".format(opt.reports_path, opt.dataset_name))
    

if __name__ == '__main__':
    main()