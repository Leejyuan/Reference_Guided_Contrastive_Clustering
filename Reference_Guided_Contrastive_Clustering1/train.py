import argparse
import os
import time
import sys
# import yaml

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR


from models.dense3D import generate_model
# from models.ResNet3D import generate_model


from models.normalize_layer import Normalize

from data.data_load import ImageDataset, get_brats2021_infer_transform,get_brats2021_train_transform
from models.mergenet_nodrop2 import TwoResNet
from NCE.NCEAverage_ref import NCEAverage_each_pre

from NCE.NCECriterion import NCECriterion, NCESoftmaxLoss, ClusterLoss
from util import AverageMeter, adjust_learning_rate, str2list
from models.linear_layer import LinearClassifier
import random
import numpy as np
from sklearn import metrics
from torch.nn import functional as F
import numpy as np
from sklearn.cluster import KMeans

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('true', '1', 'yes', 'y'):
        return True
    if value in ('false', '0', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError('Expected a boolean value: true or false.')


def save_parser_options_to_txt(options, file_path):
    with open(file_path, 'w') as f:
        for key, value in vars(options).items():
            f.write(f"{key}: {value}\n")  
            
            
def parse_option():

    parser = argparse.ArgumentParser()

    # file settings
    parser.add_argument("--data_path", type=str, required=True)
    
    parser.add_argument("--txt_file_val", type=str, required=True, help='Validation txt file: image_path label')
    parser.add_argument("--txt_file_train", type=str, required=True, help='Training txt file: image_path label')
    parser.add_argument("--txt_file_gbm_in_train", type=str, required=True, help='GBM training txt file for the reference bank')
    parser.add_argument("--txt_file_mate_in_train", type=str, required=True, help='SBM training txt file for the reference bank')
    
    parser.add_argument("--save_folder", type=str, required=True, help='Directory for experiment outputs')
    parser.add_argument("--sub_save_folder", type=str, required=True, help='Experiment-specific output subdirectory')
    
        
    # Runtime setting
    parser.add_argument("--gpu_ids", type=int, required=True)
    parser.add_argument('--data_shape', nargs=4, type=int, required=True,
                        metavar=('C', 'D', 'H', 'W'),
                        help='Input tensor shape: C D H W')
    parser.add_argument('--crop_scale', nargs=4, type=int, required=True,
                        metavar=('C', 'D', 'H', 'W'),
                        help='Crop scale: C D H W')

    parser.add_argument('--pre_train', type=str, required=True)

    # Fixed experiment configuration. These values are intentionally not exposed
    # as command-line hyperparameters to keep runs comparable.
    parser.set_defaults(
        classify_num=2,
        mode='GBM+MATE',
        train_mode='GBM+MATE',
        model='dense121_3D',
        print_freq=5,
        save_freq=5,
        val_freq=1,
        epochs=100,
        batch_size=8,
        learning_rate_base=0.003,
        learning_rate_classifier=0.003,
        lr_decay_epochs=[10, 20, 40, 50, 90],
        lr_decay_rate=[0.1, 0.5],
        momentum=0.9,
        weight_decay=1e-4,
        nce_m=0.5,
        feat_dim=128,
        pre_trained_memory=True,
        lapha=0.9,
        cluster_loss_w1=1.0,
        cluster_loss_w2=0.5,
        classify_loss_weight=1.0,
        constra_loss_weight=0.5,
        cluster_loss_weight=1.0,
        softmax=True,
        nce_k=10,
        nce_t=0.1,
    )

    opt = parser.parse_args()
    opt.save_folder = os.path.join(opt.save_folder,opt.sub_save_folder )
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    save_parser_options_to_txt(opt, os.path.join(opt.save_folder, 'parser_options.txt'))

def get_data_loader(args,txt_file_val,test_transform,mode, shuffle=True):

    dataset = ImageDataset(args.data_path, txt_file_val, test_transform, mode)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        
    )

    n_data = len(dataset)
    print('number of val samples: {}'.format(n_data))

    return data_loader, n_data

def get_pretrained_model(args, train_loader, n_data):

    model= generate_model(121,n_input_channels=args.data_shape[0],  low_dim=args.feat_dim,num_classes=args.classify_num,
                            self_supervised_features=True, classify=True,mode='ref')  
    if args.pre_train:

        ckpt = torch.load(args.pre_train, map_location=lambda storage, loc: storage)
        model_state = model.state_dict()           
        m1=ckpt['model']
        state_dict = {k:v for k,v in m1.items() if k in model_state.keys()}
        print("***********load model***************")
        print(os.path.join(args.pre_train),len(state_dict))
        model_state.update(state_dict)
        model.load_state_dict(model_state)  
        print(model.load_state_dict(model_state) )

    model.cuda(args.gpu_ids)

    model.eval()

    feat_memory=[]
    prob_memory=[]
    name_memory=[]


    for idx, (image, data_label, data_name,index) in enumerate(train_loader):
        
        if args.pre_trained_memory:
            if torch.cuda.is_available():
                image = image.cuda(args.gpu_ids,non_blocking=True)
        
            # forward
            feat, out= model(image)
            # out = classifier(feat)
            out = F.softmax(out, dim=1)
            for ind in range(len(data_name)):
        
                feat_memory.append(feat[ind,:].cpu().detach().numpy())
                prob_memory.append(out[ind][data_label[ind]].cpu().detach().numpy())
                name_memory.append(data_name[ind])
                #index_copy_(0, index.view(-1), feat.cpu().detach())
        else:
             feat = torch.rand(len(data_label), args.feat_dim)
             out = torch.ones(len(data_label),1)
             for ind in range(len(data_name)):
        
                feat_memory.append(feat[ind,:].cpu().detach().numpy())
                prob_memory.append(out[ind][data_label[ind]].cpu().detach().numpy())
                name_memory.append(data_name[ind])        
                     
    sorted_id = sorted(range(len(prob_memory)), key=lambda k: prob_memory[k], reverse=True)

    feat_memory = [feat_memory[i] for i in sorted_id]
    prob_memory = [prob_memory[i] for i in sorted_id]
    name_memory = [name_memory[i] for i in sorted_id]
    feat_memory = torch.tensor(np.array(feat_memory))
    prob_memory = torch.tensor(np.array(prob_memory))

    return feat_memory,prob_memory,name_memory

   

def set_model(args, gbm_data, mate_data, memory_gbm,memory_gbm_prob,memory_gbm_name, memory_mate,memory_mate_prob,memory_mate_name ):
    model= generate_model(121,n_input_channels=args.data_shape[0],  low_dim=args.feat_dim,num_classes=args.classify_num,
                            self_supervised_features=True, classify=True,mode='clu')  
    if args.pre_train:

        ckpt = torch.load(args.pre_train, map_location=lambda storage, loc: storage)
        model_state = model.state_dict()           
        m1=ckpt['model']
        state_dict = {k:v for k,v in m1.items() if k in model_state.keys()}
        print("***********load model***************")
        print(os.path.join(args.pre_train),len(state_dict))
        model_state.update(state_dict)
        model.load_state_dict(model_state)  
        print(model.load_state_dict(model_state) )
        
 
       
    contrast = NCEAverage_each_pre(args.feat_dim, args.gpu_ids,gbm_data, mate_data, memory_gbm,memory_gbm_prob,memory_gbm_name, memory_mate,memory_mate_prob,memory_mate_name, args.lapha, args.nce_k, args.nce_t, args.nce_m, args.softmax)
    
     
        
    criterion = NCESoftmaxLoss() if args.softmax else NCECriterion(gbm_data)

    Cluster_criterion = ClusterLoss(args.cluster_loss_w1, args.cluster_loss_w2)

    model = model.cuda(args.gpu_ids)
    # classifier = classifier.cuda()   
    l2norm = Normalize(2).cuda(args.gpu_ids)
    
    contrast = contrast.cuda(args.gpu_ids)
  
    criterion = criterion.cuda(args.gpu_ids)

    criterion_classify = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1.4])).cuda(args.gpu_ids)
   
    cudnn.benchmark = True
    
    return model, l2norm, contrast, criterion, criterion_classify, Cluster_criterion


def set_optimizer(args, model):
    
    optimizer = torch.optim.SGD(model.parameters(),  args.learning_rate_base,  momentum=args.momentum, weight_decay=args.weight_decay)

    return optimizer


@torch.no_grad()
def update_reference_banks_at_epoch_end(args, model, contrast, reference_loaders):
    """Refresh every bank entry once, after all mini-batch updates in an epoch."""
    model_was_training = model.training
    contrast_was_training = contrast.training
    model.eval()
    contrast.eval()

    for reference_loader in reference_loaders:
        for image, label, name, _ in reference_loader:
            if torch.cuda.is_available():
                image = image.cuda(args.gpu_ids, non_blocking=True)
                label = label.cuda(args.gpu_ids, non_blocking=True)

            feat, logits = model(image)
            probabilities = F.softmax(logits, dim=1)
            contrast.update_reference_banks(feat, label, probabilities, name)

    if model_was_training:
        model.train()
    if contrast_was_training:
        contrast.train()

def val(args, model, epoch,  mode ,val_loader):
    model.cuda(args.gpu_ids)
    model.eval()
    criterion_classify = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,1.1])).cuda(args.gpu_ids)
    predicts = []
    gts = []
    outputs_p = []
    outputs_logit_list = []
    Classification_losses = AverageMeter()
    print()
    with torch.no_grad():
        for i, (image, label,_, idx) in enumerate(val_loader):
            # print('***************idx*****************:',i)
            
            bsz = image.size(0)
            x = image
            y = label

            x = x.type(torch.FloatTensor).cuda(args.gpu_ids)
            y = y.cuda(args.gpu_ids).long()


            feature, outputs=  model(x)
            out_prob = nn.functional.softmax(outputs, dim=1)
            outputs_logit = outputs.argmax(dim=1)

            predicts.append(outputs_logit.cpu().detach().numpy())
            gts.append(y.cpu().detach().numpy())
            for i in range(len(out_prob.flatten().cpu().detach().numpy())//2 ):
                outputs_p.append( out_prob.flatten().cpu().detach().numpy()[i*2] )
            outputs_logit_list.append(outputs_logit.cpu().detach().numpy())
            loss_class = criterion_classify(outputs, y.long())
            Classification_losses.update(loss_class.item(), bsz)
            
    predicts = np.concatenate(predicts).flatten().astype(np.int16)
    gts = np.concatenate(gts).flatten().astype(np.int16)
    acc = metrics.accuracy_score(predicts, gts) 
    recall = metrics.recall_score(predicts, gts) 
    f1 = metrics.f1_score(predicts, gts)     
    metrics_p = metrics.confusion_matrix(predicts, gts)
    # auc = roc_auc_score(np.array(gts), np.array(outputs_p))
    ## log
    print('************************************************************************************************')
    print("epoch:"+str(epoch))
    print("acc:"+str(acc))
    print("recall:"+str(recall))
    print("f1:"+str(f1))
    print(metrics.confusion_matrix(predicts, gts))
    
    with open( os.path.join(args.save_folder,  mode+"_result.txt"),'a') as f:
        f.write('**********epoch:'+str(epoch)+'****************************\n')
        f.write(' ' + str(metrics_p))                    
        f.write(' ' + str(acc))
        f.write(' ' + str(recall)) 
        f.write(' ' + str(f1)  + '\n')
        f.write( '\n')   
        
    return acc, recall, f1, Classification_losses.avg


def train(epoch, train_loader, model, l2norm, contrast, criterion, Cluster_criterion, criterion_classify, optimizer, scheduler, args):
    """
    one epoch training
    """
    model.train()

    contrast.train()
    num_iters = 0
    losses = AverageMeter()
    Contrastive_losses = AverageMeter()
    Clustering_losses = AverageMeter()
    Classification_losses = AverageMeter()
    class_prob_meter = AverageMeter()
    print('current learn rate:',optimizer.param_groups[0]['lr'])
    if args.train_mode == "GBM+MATE":

        for idx, (image,  label,name, index) in enumerate(train_loader):
            num_iters += 1
            bsz = image.size(0)
            if torch.cuda.is_available():
                image = image.cuda(args.gpu_ids,non_blocking=True)
                label = label.cuda(args.gpu_ids,non_blocking=True)
                index = index.cuda(args.gpu_ids,non_blocking=True)
      

            # forward
            feat, out_p = model(image)
            
            out_prob = nn.functional.softmax(out_p, dim=1)

            # # output_class = classifier(feat)
          
            out_gbm_p_mate_n, feature_clu_all, gt_clu_all  = contrast( feat, label, out_prob, name,index)
            
            Clustering_loss = Cluster_criterion(feature_clu_all, gt_clu_all.long(),args.gpu_ids)
            Contrastive_loss = criterion(out_gbm_p_mate_n,args.gpu_ids)
            class_prob = out_gbm_p_mate_n[:, 0].mean()

            loss_class = criterion_classify(out_p, label.long())

            
                  
            loss=  args.classify_loss_weight*loss_class + args.constra_loss_weight* Contrastive_loss +args.cluster_loss_weight*Clustering_loss #
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # save info
            losses.update(loss.item(), bsz)
            Contrastive_losses.update(Contrastive_loss.item(), bsz)
            Clustering_losses.update(Clustering_loss.item(), bsz)
            Classification_losses.update(loss_class.item(), bsz)
      
            class_prob_meter.update(class_prob.item(), bsz)

            torch.cuda.synchronize()

            # print info
            if (idx + 1) % args.print_freq == 0:
                print('Train: [{0}][{1}/{2}]\t'
                      'loss {losses.val:.3f} ({losses.avg:.3f})\t'
                      'Contrastive loss {Contrastive_losses.val:.3f} ({Contrastive_losses.avg:.3f})\t'
                      'Clustering loss {Clustering_losses.val:.3f} ({Clustering_losses.avg:.3f})\t'
                      'Classification_loss {Classification_losses.val:.3f} ({Classification_losses.avg:.3f})\t'
                      'class_prob_meter {class_prob_meter.val:.3f} ({class_prob_meter.avg:.3f})' .format( epoch, idx + 1, len(train_loader), 
                        losses=losses, Contrastive_losses=Contrastive_losses,Clustering_losses=Clustering_losses, Classification_losses=Classification_losses, class_prob_meter=class_prob_meter))

                sys.stdout.flush()

   

    return losses.avg, Contrastive_losses.avg, Classification_losses.avg, class_prob_meter.avg

def main():
    args = parse_option()

    reference_transforms = get_brats2021_infer_transform()
    pre_train_loader_gbm, n_data_gbm = get_data_loader(
        args, args.txt_file_gbm_in_train, reference_transforms, mode='val', shuffle=False)
    pre_train_loader_mate, n_data_mate = get_data_loader(
        args, args.txt_file_mate_in_train, reference_transforms, mode='val', shuffle=False)
    
    train_transforms = get_brats2021_train_transform()
    train_loader, _ = get_data_loader(args, args.txt_file_train, train_transforms, mode = 'train')

    val_transforms = get_brats2021_infer_transform()
    val_loader, _ = get_data_loader(args,args.txt_file_val, val_transforms, mode = 'val')


    feat_GBM_memory,prob_GBM_memory ,name_GBM_memory = get_pretrained_model(args,pre_train_loader_gbm, n_data_gbm)
    feat_MATE_memory,prob_MATE_memory ,name_MATE_memory  = get_pretrained_model(args,pre_train_loader_mate, n_data_mate)

    model,l2norm, contrast, criterion, _,Cluster_criterion = set_model(args, n_data_gbm,n_data_mate, feat_GBM_memory, prob_GBM_memory ,name_GBM_memory,
                                                                                                        feat_MATE_memory,prob_MATE_memory ,name_MATE_memory)  #classifier, 
    # set the optimizer
    optimizer = set_optimizer(args, model)
    # tensorboard
    writer = SummaryWriter(log_dir=args.save_folder)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5) #scheduler.step(val_loss)
    # routine
    train_Classification_acc_pre = 0
    test_Classification_acc_pre = 0
    for epoch in range(0, args.epochs+1):
        adjust_learning_rate(epoch, args, optimizer)

        criterion_classify = nn.CrossEntropyLoss(weight=torch.FloatTensor(args.ce_w)).cuda(args.gpu_ids)


        print("==> training...")
        losses, Contrastive_losses, Classification_losses, class_prob_meter= train(epoch, train_loader, model,  l2norm, contrast, criterion,Cluster_criterion, criterion_classify, optimizer, scheduler, args)

        print("==> updating reference banks at epoch end...")
        update_reference_banks_at_epoch_end(
            args, model, contrast, [pre_train_loader_gbm, pre_train_loader_mate])

        # tensorboard logger
        writer.add_scalar('losses', losses, epoch)
        writer.add_scalar('Contrastive_losses', Contrastive_losses, epoch)
        writer.add_scalar('Classification_losses', Classification_losses, epoch)
        writer.add_scalar('class_prob_meter', class_prob_meter, epoch)
 
        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                # 'classifier': classifier.state_dict(),
                'contrast': contrast.state_dict(),

                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(args.save_folder,'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)          
            
        torch.cuda.empty_cache()
        if epoch % args.val_freq == 0:
            print('==> Testing...')
            val_Classification_acc, _, _, val_Classification_losses = val(args, model, epoch,'rest', val_loader) 
           
            writer.add_scalar('test_Classification_losses', val_Classification_losses, epoch)
            writer.add_scalar('test_Classification_acc', val_Classification_acc, epoch)
            
        torch.cuda.empty_cache()
        if val_Classification_acc >= val_Classification_acc_pre :
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                # 'classifier': classifier.state_dict(),
                'contrast': contrast.state_dict(),

                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            save_file = os.path.join(args.save_folder,'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)    

        if  val_Classification_acc >= val_Classification_acc_pre: 
            val_Classification_acc_pre = val_Classification_acc


if __name__ == '__main__':
    main()
