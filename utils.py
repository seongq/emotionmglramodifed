import numpy as np, argparse, time, random
import torch
def seed_everything(dataset_name):
    if dataset_name =="IEMOCAP":
        seed =1475
    elif dataset_name == "MELD":
        seed = 67137
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report

def compute_detailed_metrics(labels, preds, sample_weight=None):
    report = classification_report(labels, preds, sample_weight=sample_weight, output_dict=True)
    cm = confusion_matrix(labels, preds, sample_weight=sample_weight)
    
    # class-wise accuracy
    class_acc = (cm.diagonal() / cm.sum(axis=1)).tolist()
    
    # class-wise f1
    class_f1 = [report[str(i)]['f1-score'] for i in range(len(class_acc))]

    # weighted accuracy is just accuracy_score with sample_weight
    weighted_accuracy = accuracy_score(labels, preds, sample_weight=sample_weight)

    return {
        'class_accuracy': class_acc,
        'class_f1': class_f1,
        'weighted_accuracy': weighted_accuracy
    }