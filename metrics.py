import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score, classification_report, confusion_matrix)

class ComputeMetrics():
    def __init__(self, cls_mode, model, data_loader, device, average='macro', dataset='nb15'):
        self.cls_mode = cls_mode
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.average = average
        self.pred_results()
        
        if self.cls_mode == 'multiclass':
            if dataset == 'ids12':
                self.target_names = ['Benign', 'Bot', 'Infiltration', 'DoS', 'Brute force']
            elif dataset == 'nb15':
                self.target_names = ['Normal', 'Exploits', 'Fuzzers', 'Reconnaissance', 'Generic', 'DoS', 'Shellcode', 'Analysis', 'Backdoor', 'Worms']
            elif dataset == 'ids17':
                self.target_names = ['Benign', 'DoS', 'PortScan', 'DDoS', 'Patator', 'Web Attack', 'Bot', 'Infiltration']
        else:
            if dataset == 'ids12':
                self.target_names = ['Benign', 'Attack']
            elif dataset == 'nb15':
                self.target_names = ['Normal', 'Attack']
            elif dataset == 'ids17':
                self.target_names = ['Benign', 'Attack']
               

    def pred_results(self):
        self.model.eval()
        self.predicted = np.array([])
        self.excepted = np.array([])
        
        with torch.set_grad_enabled(False):
            for data, labels in self.data_loader:
                data = data.to(self.device)
                
                if self.cls_mode == 'multiclass':
                    labels = labels.to(self.device)
                else:
                    one = torch.ones_like(labels)
                    labels = torch.where(labels > 0, one, labels)
                    labels = labels.to(self.device)
                
                pred = self.model(data)
                
                probas = F.softmax(pred, dim=1)
                _, pred_labels = torch.max(probas, 1)
                
                self.excepted = np.append(self.excepted, labels.cpu().numpy(), axis = 0)
                self.predicted = np.append(self.predicted, pred_labels.cpu().numpy(), axis = 0)

    def accuracy(self):
        accuracy_ = accuracy_score(self.excepted, self.predicted)
        return accuracy_
    def precision(self):
        precision_ = precision_score(self.excepted, self.predicted, average=self.average) 
        return precision_
    def recall(self):
        recall_ = recall_score(self.excepted, self.predicted, average=self.average) 
        return recall_
    def f1(self):
        f1_ = f1_score(self.excepted, self.predicted, average=self.average) 
        return f1_
    def cr(self):
        cr_ = classification_report(self.excepted, self.predicted, target_names= self.target_names, digits=4)
        return cr_
    def cm(self):
        cm_ = confusion_matrix(self.excepted, self.predicted)
        return cm_
    

    

