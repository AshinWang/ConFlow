import time
import os,shutil,math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
import torch.nn.functional as F
from loss_supcon import SupConLoss
from metrics import ComputeMetrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score, classification_report, confusion_matrix)


def training(model, 
            train_loader, 
            valid_loader,
            loss_fn='scl+ce', # 'ce'
            cls_mode='multiclass', # 'binary'
            epochs=500, 
            learning_rate=5e-5,
            early_stop=100,
            temperature=0.3,
            alpha=0.9,
            dataset='ids17',
            weight_decay=1e-6,
            ):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    supcon_loss = SupConLoss(temperature=temperature)
    cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
    

    def _train_epoch(loss_fn, cls_mode, model, train_loader, optimizer, alpha):
        loss_meter = AvgMeter()
        tqdm_object = tqdm(train_loader, total=len(train_loader))
        model.train()
        for idx, (data, labels) in enumerate(tqdm_object):
            bsz = labels.size(0)
            data = data.to(device)
            if cls_mode == 'multiclass':
                labels = labels.to(device)
            else:
                one = torch.ones_like(labels)
                labels = torch.where(labels > 0, one, labels)
                labels = labels.to(device)
            
            if loss_fn == 'scl+ce':
                f1 = model.encoder(data)
                f2 = model.encoder(data)

                g1 = model.classifier(f1)
                g2 = model.classifier(f2)

                z1 = model.projection(f1)
                z2 = model.projection(f2)
            
                features = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)

                sc_loss = supcon_loss(features, labels)
                ce_loss1 = cross_entropy_loss(g1, labels)
                ce_loss2 = cross_entropy_loss(g2, labels)
                ce_loss = (ce_loss1.mean() + ce_loss2.mean()).mean()
                loss = (1 - alpha) * ce_loss + alpha*sc_loss 
            
            elif loss_fn == 'ce':
                f = model(data)
                ce_loss = cross_entropy_loss(f, labels)
                loss = ce_loss.mean()
            else:
                raise ValueError('Unsupported mode!')
                    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), bsz)
            tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
        return loss_meter
    
    def _valid_epoch(loss_fn, cls_mode, model, valid_loader, optimizer, alpha):
        loss_meter = AvgMeter()
        tqdm_object = tqdm(valid_loader, total=len(valid_loader))
        model.eval()
        with torch.no_grad():
            for data, labels in tqdm_object:
                bsz = labels.size(0)
                data = data.to(device)
                
                if cls_mode == 'multiclass':
                    labels = labels.to(device)
                else:
                    one = torch.ones_like(labels)
                    labels = torch.where(labels > 0, one, labels)
                    labels = labels.to(device)

                if loss_fn == 'scl+ce':
                    f1 = model.encoder(data)
                    f2 = model.encoder(data)

                    g1 = model.classifier(f1)
                    g2 = model.classifier(f2)

                    z1 = model.projection(f1)
                    z2 = model.projection(f2)

                    features = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)

                    sc_loss = supcon_loss(features, labels)
                    ce_loss1 = cross_entropy_loss(g1, labels)
                    ce_loss2 = cross_entropy_loss(g2, labels)
                    ce_loss = (ce_loss1.mean() + ce_loss2.mean()).mean()
                    loss = (1 - alpha) * ce_loss + alpha*sc_loss 

                elif loss_fn == 'ce':
                    g = model(data)
                    ce_loss = cross_entropy_loss(g, labels)
                    loss = ce_loss.mean()
                else:
                    raise ValueError('Unsupported mode!')
                
                loss_meter.update(loss.item(), data.size(0))
                tqdm_object.set_postfix(valid_loss=loss_meter.avg)
        return loss_meter

    
    best_train_loss = float('inf')
    best_valid_loss = float('inf')
    best_acc = 0
    
    train_losses = []
    valid_losses = []
    v_f1s = []
    v_accs = []
    epochss = []
    time_now = time.time()
    log_path = mk_dir('logs/{}'.format(dataset))
    ckpt_path = mk_dir('ckpt/{}'.format(dataset))
    
    
    
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        train_loss = _train_epoch(loss_fn, cls_mode, model, train_loader, optimizer, alpha)
        valid_loss = _valid_epoch(loss_fn, cls_mode, model, valid_loader, optimizer, alpha)
        
        val_metrics = ComputeMetrics(cls_mode, model, valid_loader, device, 'macro', dataset)
        v_f1 = val_metrics.f1()
        v_acc = val_metrics.accuracy()
        print('| Valid set F1-Score: {},  Accuracy: {} |'.format(v_f1, v_acc))
        
        train_losses.append(train_loss.avg)
        valid_losses.append(valid_loss.avg)
        v_accs.append(v_acc)
        v_f1s.append(v_f1)
        epochss.append(epoch)
        
        logs = {'epoch': epochss,
                'train_loss': train_losses,
                'valid_loss': valid_losses,
                'valid_f1': v_f1,
                'valid_acc': v_acc,
                }
        
        df = pd.DataFrame(logs)
        df.to_csv('{}/log_{}.csv'.format(log_path, time_now))
        
        if train_loss.avg < best_train_loss:
            best_train_loss = train_loss.avg
            torch.save(model.state_dict(), "{}/best_train_loss.pt".format(ckpt_path))
            print("Saved best train loss: {}".format(best_train_loss))
            
        
        if valid_loss.avg < best_valid_loss:
            best_valid_loss = valid_loss.avg
            torch.save(model.state_dict(), "{}/best_valid_loss.pt".format(ckpt_path))
            print("Saved Best valid loss: {}".format(best_valid_loss))
            idx = 0
            
            
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), "{}/best_acc.pt".format(ckpt_path))
            print("Saved Best Valid Acc: {}".format(best_acc))
            idx = 0
            
        else:
            idx += 1
            if idx >= early_stop:
                print("Early stopping!", epoch)
                break
        
    return model 


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def mk_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path