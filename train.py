import os
import pickle

import numpy as np, argparse, time, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, LSTMModel, GRUModel, Model, MaskedMSELoss, FocalLoss
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import pickle as pk
import datetime
import torch.nn.functional as F
from utils import seed_everything,compute_detailed_metrics

# seed = 67137 # We use seed = 1475 on IEMOCAP and seed = 67137 on MELD


def get_train_valid_sampler(trainset, valid=0.1, dataset='IEMOCAP'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])



def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader
    

def train_or_eval_graph_model(model, loss_function, dataloader, epoch, cuda, modals, optimizer=None, train=False, dataset='IEMOCAP'):
    losses, preds, labels = [], [], []
    vids = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything(seed_number)
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        
        textf1,textf2,textf3,textf4, visuf, acouf, qmask, umask, label = [d.to(device) for d in data[:-2]] if cuda else data[:-2]
        
        if args.multi_modal:
            if args.mm_fusion_mthd=='concat':
                if modals == 'avl':
                    textf = torch.cat([acouf, visuf, textf1,textf2,textf3,textf4],dim=-1)
                elif modals == 'av':
                    textf = torch.cat([acouf, visuf],dim=-1)
                elif modals == 'vl':
                    textf = torch.cat([visuf, textf1,textf2,textf3,textf4],dim=-1)
                elif modals == 'al':
                    textf = torch.cat([acouf, textf1,textf2,textf3,textf4],dim=-1)
                else:
                    raise NotImplementedError
            elif args.mm_fusion_mthd=='gated':
                textf = textf
        else:
            if modals == 'a':
                textf = acouf
            elif modals == 'v':
                textf = visuf
            elif modals == 'l':
                textf = textf
            else:
                raise NotImplementedError

        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]

        if args.multi_modal and args.mm_fusion_mthd=='gated':
            log_prob  = model(textf, qmask, umask, lengths, acouf, visuf)
        elif args.multi_modal and args.mm_fusion_mthd=='concat_subsequently':   
            log_prob = model([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
        elif args.multi_modal and args.mm_fusion_mthd=='concat_DHT':   
            log_prob = model([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch)
        else:
            log_prob = model(textf, qmask, umask, lengths)
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        loss = loss_function(log_prob, label)
        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())
        if train:
            loss.backward()
            optimizer.step()

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan')

    vids += data[-1]
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds)*100, 2)
    avg_fscore = round(f1_score(labels,preds, average='weighted')*100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids


if __name__ == '__main__':
    path = './saved/IEMOCAP/'

    parser = argparse.ArgumentParser()

    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')

    parser.add_argument('--base-model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')

    parser.add_argument('--graph-model', action='store_true', default=True, help='whether to use graph model after recurrent encoding')

    parser.add_argument('--nodal-attention', action='store_true', default=True, help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    
    parser.add_argument('--l2', type=float, default=0.00003, metavar='L2', help='L2 regularization weight')
    
    parser.add_argument('--rec-dropout', type=float, default=0.1, metavar='rec_dropout', help='rec_dropout rate')
    
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    
    parser.add_argument('--batch-size', type=int, default=16, metavar='BS', help='batch size')
    
    parser.add_argument('--epochs', type=int, default=300, metavar='E', help='number of epochs')
    
    parser.add_argument('--class-weight', action='store_true', default=True, help='use class weights')
    
    parser.add_argument('--active-listener', action='store_true', default=False, help='active listener')
    
    parser.add_argument('--attention', default='general', help='Attention type in DialogRNN model')
    
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')

    parser.add_argument('--graph_type', default='hyper', help='relation/GCN3/DeepGCN/MMGCN/MMGCN2')

    parser.add_argument('--use_topic', action='store_true', default=False, help='whether to use topic information')

    parser.add_argument('--alpha', type=float, default=0.2, help='alpha')

    parser.add_argument('--multiheads', type=int, default=6, help='multiheads')

    parser.add_argument('--graph_construct', default='full', help='single/window/fc for MMGCN2; direct/full for others')

    parser.add_argument('--use_gcn', action='store_true', default=False, help='whether to combine spectral and none-spectral methods or not')

    parser.add_argument('--use_residue', action='store_true', default=False, help='whether to use residue information or not')

    parser.add_argument('--multi_modal', action='store_true', default=True, help='whether to use multimodal information')

    parser.add_argument('--mm_fusion_mthd', default='concat_DHT', help='method to use multimodal information: concat, gated, concat_subsequently')

    parser.add_argument('--modals', default='avl', help='modals to fusion')

    parser.add_argument('--av_using_lstm', action='store_true', default=False, help='whether to use lstm in acoustic and visual modality')

    parser.add_argument('--Deep_GCN_nlayers', type=int, default=4, help='Deep_GCN_nlayers')

    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')

    parser.add_argument('--use_speaker', action='store_true', default=True, help='whether to use speaker embedding')

    parser.add_argument('--use_modal', action='store_true', default=False, help='whether to use modal embedding')

    parser.add_argument('--norm', default='BN', help='NORM type')

    parser.add_argument('--testing', action='store_true', default=False, help='testing')

    parser.add_argument('--num_L', type=int, default=3, help='num_hyperconvs')

    parser.add_argument('--num_K', type=int, default=4, help='num_convs')
    
    parser.add_argument("--seed_number", type=int, default=1, required=True)

    parser.add_argument("--self_attention", action="store_true", default=False)
    
    parser.add_argument("--original_gcn", default=False, action="store_true")
    
    parser.add_argument("--graph_masking", default=True, action="store_false")
    args = parser.parse_args()
    today = datetime.datetime.now()
    print(args)
    if args.av_using_lstm:
        name_ = args.mm_fusion_mthd+'_'+args.modals+'_'+args.graph_type+'_'+args.graph_construct+'using_lstm_'+args.Dataset
    else:
        name_ = args.mm_fusion_mthd+'_'+args.modals+'_'+args.graph_type+'_'+args.graph_construct+str(args.Deep_GCN_nlayers)+'_'+args.Dataset
        print(name_)

    if args.use_speaker:
        name_ = name_+'_speaker'
    if args.use_modal:
        name_ = name_+'_modal'

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        device = torch.device("cuda:0")
        print('Running on GPU')
    else:
        device = torch.device("cpu")
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size
    modals = args.modals
    seed_number = args.seed_number
    feat2dim = {'IS10':1582,'3DCNN':512,'textCNN':100,'bert':768,'denseface':342,'MELD_text':600,'MELD_audio':300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024 #feat2dim['textCNN'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_text']

    if args.multi_modal:
        if args.mm_fusion_mthd=='concat':
            if modals == 'avl':
                D_m = D_audio+D_visual+D_text
            elif modals == 'av':
                D_m = D_audio+D_visual
            elif modals == 'al':
                D_m = D_audio+D_text
            elif modals == 'vl':
                D_m = D_visual+D_text
            else:
                raise NotImplementedError
        else:
            D_m = 1024
    else:
        if modals == 'a':
            D_m = D_audio
        elif modals == 'v':
            D_m = D_visual
        elif modals == 'l':
            D_m = D_text
        else:
            raise NotImplementedError
    D_g = 512 if args.Dataset=='IEMOCAP' else 1024
    D_p = 150
    D_e = 100
    D_h = 100
    D_a = 100
    graph_h = 512
    n_speakers = 9 if args.Dataset=='MELD' else 2
    n_classes  = 7 if args.Dataset=='MELD' else 6 if args.Dataset=='IEMOCAP' else 1


    if args.graph_model:
        seed_everything(seed_number)
        print(args.original_gcn)
        print(args.graph_masking)
        model = Model(args.base_model,
                                 D_m, D_g, D_e, graph_h,
                                 n_speakers=n_speakers,
                                 n_classes=n_classes,
                                 dropout=args.dropout,
                                 no_cuda=args.no_cuda,
                                 graph_type=args.graph_type,
                                 use_topic=args.use_topic,
                                 alpha=args.alpha,
                                 multiheads=args.multiheads,
                                 graph_construct=args.graph_construct,
                                 use_GCN=args.use_gcn,
                                 use_residue=args.use_residue,
                                 D_m_v = D_visual,
                                 D_m_a = D_audio,
                                 modals=args.modals,
                                 att_type=args.mm_fusion_mthd,
                                 av_using_lstm=args.av_using_lstm,
                                 dataset=args.Dataset,
                                 use_speaker=args.use_speaker,
                                 use_modal=args.use_modal,
                                 num_L = args.num_L,
                                 num_K = args.num_K,
                                 modality_self_attention = args.self_attention,
                                 original_gcn= args.original_gcn,
                                 graph_masking=args.graph_masking)

        print ('Graph NN with', args.base_model, 'as base model.')
        name = 'Graph'

    if cuda:
        model.to(device)

    if args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])

    if args.Dataset == 'MELD':
        loss_function = FocalLoss()
    else:
        if args.class_weight:
            if args.graph_model:
                #loss_function = FocalLoss()
                loss_function  = nn.NLLLoss(loss_weights.to(device) if cuda else loss_weights)
            else:
                loss_function  = MaskedNLLLoss(loss_weights.to(device) if cuda else loss_weights)
        else:
            if args.graph_model:
                loss_function = nn.NLLLoss()
            else:
                loss_function = MaskedNLLLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

    lr = args.lr
    
    if args.Dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0,
                                                                    batch_size=batch_size,
                                                                    num_workers=2)
    elif args.Dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0,
                                                                      batch_size=batch_size,
                                                                      num_workers=2)
    else:
        print("There is no such dataset")

    best_fscore, best_acc, best_loss, best_label_f1, best_label_acc, best_pred_f1, best_pred_acc , best_mask = -1000, -1000, None, None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []


# original_gcn", default=False, action="store_true")
    
#     parser.add_argument("--graph_masking"    
    if args.av_using_lstm:
        model_save_dir = os.path.join("/workspace/MGLRA/save_folder", args.Dataset, f"original_av_using_lstm_self_attention_{args.self_attention}_graphmasking_{args.graph_masking}_originalgcn_{args.original_gcn}")
        os.makedirs(model_save_dir, exist_ok=True)
    else:
        model_save_dir = os.path.join("/workspace/MGLRA/save_folder", args.Dataset, f"original_self_attention_{args.self_attention}_graphmasking_{args.graph_masking}_originalgcn_{args.original_gcn}")
        os.makedirs(model_save_dir, exist_ok=True)
    
    from datetime import datetime
    import pytz

    kst = pytz.timezone("Asia/Seoul")
    now_kst = datetime.now(kst)
    timestamp_str = now_kst.strftime("%Y%m%d%H%M")
    # print(timestamp_str)
    pickle_path = os.path.join(f"/workspace/MGLRA/result/{args.Dataset}", "result.pkl")
    best_f1_model_path = None
    best_acc_model_path = None
    for e in range(n_epochs):
        print(f"epoch: {e}")
        start_time = time.time()

        train_loss, train_acc, _, _, train_fscore, _ = train_or_eval_graph_model(model,  loss_function, train_loader, e, cuda, args.modals, \
                                                                                 optimizer, True, dataset=args.Dataset)
        valid_loss, valid_acc, _, _, valid_fscore = train_or_eval_graph_model(model,  loss_function, valid_loader, e, cuda, args.modals, \
                                                                              dataset=args.Dataset,)
        test_loss, test_acc, test_label, test_pred, test_fscore, _ = train_or_eval_graph_model(model, loss_function, test_loader, e, cuda, args.modals, \
                                                                                               dataset=args.Dataset)
        all_fscore.append(test_fscore)
        all_acc.append(test_acc)

        if (best_fscore < test_fscore):
            print("fscore 높네")
            best_fscore = test_fscore
            best_acc = test_acc
            best_label_f1, best_pred_f1 = test_label, test_pred
            
            best_mask = None  # ensure mask is available

            # metrics 계산
            f1_metrics = compute_detailed_metrics(best_label_f1, best_pred_f1, sample_weight=best_mask)
            wf1 = f1_score(best_label_f1, best_pred_f1, sample_weight=best_mask, average='weighted')
            wacc = f1_metrics['weighted_accuracy']
            acc = accuracy_score(best_label_f1, best_pred_f1)
            
            class_accuracy = f1_metrics["class_accuracy"]
            class_f1 = f1_metrics["class_f1"]
            weighted_accuracy = f1_metrics['weighted_accuracy']
            weighted_f1 = f1_metrics['weighted_f1']
            
            result_dictionary = {}
            result_dictionary['f1_w'] = weighted_f1
            result_dictionary['acc_w'] = weighted_accuracy
            result_dictionary['seed'] = seed_number
            result_dictionary['timestamps'] = timestamp_str
            for i in range(len(class_accuracy)):
                result_dictionary[f"acc_{i}"] = class_accuracy[i]
            
            for i in range(len(class_f1)):
                result_dictionary[f"f1_{i}"]= class_f1[i]
                
            if args.av_using_lstm:
                result_dictionary["mode"]=f"ORIGINAL_AV_LSTM_self_attention_{args.self_attention}_graph_masking_{args.graph_masking}_originalgcn_{args.original_gcn}"
            else:
                result_dictionary["mode"]=f"ORIGINAL_self_attention_{args.self_attention}_graph_masking_{args.graph_masking}_originalgcn_{args.original_gcn}"
            
            # print(result_dictionary)
            # print(weighted_accuracy == acc)
            # print(weighted_f1 == wf1)

            filename = f"model_f1_{best_fscore:.2f}_acc_{acc*100:.2f}_epoch_{e}_time_{timestamp_str}_seed_{seed_number}.pth"
            
            if best_f1_model_path==None:
                # print("여긴데")
                best_f1_model_path = os.path.join(model_save_dir, filename)
            else:
                # print("이거는")
                # print(best_f1_model_path)
                os.remove(str(best_f1_model_path))
                best_f1_model_path = os.path.join(model_save_dir, filename)
            
            result_dictionary['path'] = best_f1_model_path
            result_dictionary['epoch'] = e
            
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    tempdata = pickle.load(f)
                    for key in tempdata.keys():
                        tempdata[key].append(result_dictionary[key])
            else:
                tempdata = {}
                for key in result_dictionary.keys():
                    tempdata[key] = [result_dictionary[key]]
                    
            # print(tempdata)
                
                
            with open(pickle_path, "wb") as f:
                pickle.dump(tempdata, f)

            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "metrics": f1_metrics,
                "timestamps": timestamp_str,
                "seed": seed_number,
            }, best_f1_model_path)
                
    
            if args.tensorboard:
                writer.add_scalar('test: accuracy', test_acc, e)
                writer.add_scalar('test: fscore', test_fscore, e)
                writer.add_scalar('train: accuracy', train_acc, e)
                writer.add_scalar('train: fscore', train_fscore, e)

            print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                    format(e+1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
            
        elif (best_acc < test_acc):
            print("accuracy가 높네")
            best_fscore = test_fscore
            best_acc = test_acc
            best_label_f1, best_pred_f1 = test_label, test_pred
            
            best_mask = None  # ensure mask is available

            # metrics 계산
            f1_metrics = compute_detailed_metrics(best_label_f1, best_pred_f1, sample_weight=best_mask)
            wf1 = f1_score(best_label_f1, best_pred_f1, sample_weight=best_mask, average='weighted')
            wacc = f1_metrics['weighted_accuracy']
            acc = accuracy_score(best_label_f1, best_pred_f1)
            
            class_accuracy = f1_metrics["class_accuracy"]
            class_f1 = f1_metrics["class_f1"]
            weighted_accuracy = f1_metrics['weighted_accuracy']
            weighted_f1 = f1_metrics['weighted_f1']
            
            result_dictionary = {}
            result_dictionary['f1_w'] = weighted_f1
            result_dictionary['acc_w'] = weighted_accuracy
            result_dictionary['seed'] = seed_number
            result_dictionary['timestamps'] = timestamp_str
            for i in range(len(class_accuracy)):
                result_dictionary[f"acc_{i}"] = class_accuracy[i]
            
            for i in range(len(class_f1)):
                result_dictionary[f"f1_{i}"]= class_f1[i]
                
            if args.av_using_lstm:
                result_dictionary["mode"]=f"ORIGINAL_AV_LSTM_self_attention_{args.self_attention}_graph_masking_{args.graph_masking}_originalgcn_{args.original_gcn}"
            else:
                result_dictionary["mode"]=f"ORIGINAL_self_attention_{args.self_attention}__graph_masking_{args.graph_masking}_originalgcn_{args.original_gcn}"
            
            # print(result_dictionary)
            # print(weighted_accuracy == acc)
            # print(weighted_f1 == wf1)

            filename = f"model_f1_{best_fscore:.2f}_acc_{acc*100:.2f}_epoch_{e}_time_{timestamp_str}_seed_{seed_number}.pth"
            
            if best_acc_model_path==None:
                # print("여긴데")
                best_acc_model_path = os.path.join(model_save_dir, filename)
            else:
                # print("이거는")
                # print(best_acc_model_path)
                os.remove(str(best_acc_model_path))
                best_acc_model_path = os.path.join(model_save_dir, filename)
            
            result_dictionary['path'] = best_acc_model_path
            result_dictionary['epoch'] = e
            
            if os.path.exists(pickle_path):
                with open(pickle_path, "rb") as f:
                    tempdata = pickle.load(f)
                    for key in tempdata.keys():
                        tempdata[key].append(result_dictionary[key])
            else:
                tempdata = {}
                for key in result_dictionary.keys():
                    tempdata[key] = [result_dictionary[key]]
                    
            # print(tempdata)
                
                
            with open(pickle_path, "wb") as f:
                pickle.dump(tempdata, f)

            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "metrics": f1_metrics,
                "timestamps": timestamp_str,
                "seed": seed_number,
            }, best_acc_model_path)
                
    
            if args.tensorboard:
                writer.add_scalar('test: accuracy', test_acc, e)
                writer.add_scalar('test: fscore', test_fscore, e)
                writer.add_scalar('train: accuracy', train_acc, e)
                writer.add_scalar('train: fscore', train_fscore, e)

            print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                    format(e+1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
        # if (e+1)%10 == 0:
        #     print ('----------best F-Score:', max(all_fscore))
        #     print(classification_report(best_label_f1, best_pred_f1, sample_weight=best_mask,digits=4))
        #     print(confusion_matrix(best_label_f1,best_pred_f1,sample_weight=best_mask))
            
        #     print ('----------best acc:', max(all_acc))
        #     print(classification_report(best_label_acc, best_pred_acc, sample_weight=best_mask,digits=4))
        #     print(confusion_matrix(best_label_acc,best_pred_acc,sample_weight=best_mask))


        
    

    if args.tensorboard:
        writer.close()
    if not args.testing:
        print('Test performance.. by F-score')
        print ('F-Score:', max(all_fscore))

        print(classification_report(best_label_f1, best_pred_f1, sample_weight=best_mask,digits=4))
        print(confusion_matrix(best_label_f1,best_pred_f1,sample_weight=best_mask))
        print("=========학습끝") 