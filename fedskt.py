import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.utils import save_image
import torchvision.transforms as transforms
import numpy as np
from autoencoder_pretrained import create_autoencoder
from utils import KL_Loss,CE_Loss
from torch import nn
import utils
import torch.nn.functional as F
from math import exp
import torch.utils.data as data
from model_zoo import create_model
import random

def test_on_cloud(cloud_model,test_data_global,comm_round):
    acc_all=[]  
    if True:
        loss_avg = utils.RunningAverage()
        accTop1_avg = utils.RunningAverage() 
        accTop5_avg = utils.RunningAverage()
        for batch_idx, (images, labels) in enumerate(test_data_global): 
            images, labels = images.cuda(), labels.cuda()
            labels=torch.tensor(labels,dtype=torch.long) 

            log_probs, extracted_features = cloud_model(images) 
            metrics = utils.accuracy(log_probs, labels, topk=(1, 5)) 
            accTop1_avg.update(metrics[0].item()) 
            accTop5_avg.update(metrics[1].item())
        test_metrics = {
                        'test_accTop1': accTop1_avg.value(),
                        'test_accTop5': accTop5_avg.value(),
                        }
        print("Test/AccTop1 in comm_round",comm_round,test_metrics['test_accTop1'])

def run_fedskt(client_models,edge_models,cloud_model,train_data_local_num_dict, test_data_local_num_dict,train_data_local_dict, test_data_local_dict, test_data_global, train_data_global,args):
    V1=[Node(args,cloud_model)] 
    V2=create_child_for_upper_level(args,V1,args.edge_number,edge_models) 
    assert args.client_number%args.edge_number==0
    V3=create_child_for_upper_level(args,V2,args.client_number//args.edge_number,iter(client_models)) 
    for idx,node in enumerate(V3):
        node.dataset=[_ for _ in train_data_local_dict[idx]] 
    Init(V1[0])

    for comm_round in range(args.comm_round):
        if comm_round == 0:
            for i in range(1):
                print('------------------',i,'----------------------')
        else :
            train_FedSkt(V1[0],args)
            test_on_cloud(V1[0].model,test_data_global,comm_round)




global_index=0
class Node:
    def __init__(self, args, model, father=None):
         global global_index
         self.model=model.cuda()
         self.dataset=None
         self.father = father 
         self.children = [] 
         self.index=global_index
         self.autoencoder=create_autoencoder().cuda() 
         self.noises=[]
         self.labels=[]
         self.args=args
         self.is_teacher=False
         global_index=global_index+1 
    
    def is_leaf(self):
        return len(self.children)==0

    def is_root(self):
        return self.father==None

    def __repr__(self):
        return '[index:'+str(self.index)+' data:'+str(self.data)+' father:'+str(self.father.index)+\
            ' +children:'+(str([_.index for _ in self.children]) if len(self.children)!=0 else "[]")+']'

def create_child_for_upper_level(args,upper_level,children_number,models):
    result=[] #
    for ele in upper_level: 
        sub_nodes=[] 
        for _,model in zip(range(children_number),models):
            sub_nodes.append(Node(args,model,ele))
        ele.children=sub_nodes 
        result.extend(sub_nodes) 
    return result


def train_FedSkt(node,args):
    r = args.r
    if node.is_root():
        for child in node.children:
            train_FedSkt(child,args)
    elif node.is_leaf():
        for i in range(r):
            BSBODP(node,node.father,args)
    else:
        for child in node.children:
            train_FedSkt(child,args)
        for i in range(r):
            BSBODP(node,node.father,args)


def Init(node):
    if node.is_root(): 
        for child in node.children:
            Init(child) 
    elif node.is_leaf(): 
        for idx,data in enumerate(node.dataset):
            img,label=data 
            img,label=img.cuda(),label.cuda() 
            noise=node.autoencoder.encoder(img) 
            node.noises.append(noise) 
            node.labels.append(label) 
        node.father.noises.extend(node.noises) 
        node.father.labels.extend(node.labels) 
    else: 
        for child in node.children:
            Init(child) 
        node.father.noises.extend(node.noises) 
        node.father.labels.extend(node.labels) 


        
def BSBODP(node1,node2,args): 
    if node1.is_teacher:
        BSBODP_TS(node2,node1,args) 
    else:
        BSBODP_TS(node1,node2,args) 


class Loss_Non_Leaf(nn.Module): 
    def __init__(self, temperature=3, alpha=10,alpha2=1): 
        super(Loss_Non_Leaf, self).__init__()
        self.alpha = alpha
        self.alpha2 = alpha2
        self.kl_loss_crit=KL_Loss(temperature) 
        self.ce_loss_crit=nn.CrossEntropyLoss() 

    def forward(self, output_batch, teacher_outputs, label):
        loss_ce=self.ce_loss_crit(output_batch,label.long()) 
        loss_kl=self.kl_loss_crit(output_batch,teacher_outputs.detach()) 
        loss_true=self.alpha2 * loss_ce+self.alpha*loss_kl 
        return loss_true


class Loss_Leaf(nn.Module): 
    def __init__(self, temperature=3, alpha=10, alpha2=1): 
        super(Loss_Leaf, self).__init__()
        self.alpha = alpha
        self.alpha2 = alpha2
        self.non_leaf_loss_crit=Loss_Non_Leaf(temperature,alpha) 
        self.ce_loss_crit=nn.CrossEntropyLoss() 
    def forward(self, output_fake, teacher_outputs_fake, output_true, label):
        loss_leaf=self.non_leaf_loss_crit(output_fake, teacher_outputs_fake.detach(), label.long())
        loss_ce=self.ce_loss_crit(output_true,label.long())
        loss_true=1 * loss_leaf+1 * loss_ce 
        return loss_true


def update_teacher(node_origin,node_neigh,noises,labels,args):
    accTop1_teacher = utils.RunningAverage()
    accTop1_student = utils.RunningAverage()
    for idx,(noise,label) in enumerate(zip(noises,labels)):

        fake_data=node_neigh.autoencoder.decoder(noise)

        nei_logits,_=node_neigh.model(fake_data)
        logits_fake,_=node_origin.model(fake_data)
        metrics1 = utils.accuracy_test(nei_logits, label, topk=(1, 2))
        metrics2 = utils.accuracy_test(logits_fake, label, topk=(1, 2))
        accTop1_teacher.update(metrics1[0].item())
        accTop1_student.update(metrics2[0].item())
    teacher_metrics = {
                       'test_accTop1': accTop1_teacher.value(),
                        }
    student_metrics = {
                        'test_accTop1': accTop1_student.value(),
                     }

    if node_origin.index > args.edge_number:
        if teacher_metrics['test_accTop1'] < student_metrics['test_accTop1']:
            node_origin.is_teacher = True
    if node_neigh.index > args.edge_number:
        if teacher_metrics['test_accTop1'] < student_metrics['test_accTop1']:
            node_neigh.is_teacher = False
    if node_origin.index == 0:
        if teacher_metrics['test_accTop1'] < student_metrics['test_accTop1']:
            node_neigh.is_teacher = False
    if node_neigh.index == 0:
        if teacher_metrics['test_accTop1'] < student_metrics['test_accTop1']:
            node_origin.is_teacher = True


def data_selection(noises,labels,s):
    all_data=0
    for idx,(noise,label) in enumerate(zip(noises,labels)):
        all_data=all_data+1

    select_num=int(all_data/s)
    select_data = random.sample(range(all_data), select_num)
    return select_data


def BSBODP_TS(node_origin,node_neigh,args):
    noises=node_neigh.noises if len(node_neigh.noises)<len(node_origin.noises) else node_origin.noises
    labels=node_neigh.labels if len(node_neigh.labels)<len(node_origin.labels) else node_origin.labels
    crit_non_leaf=Loss_Non_Leaf(args.T_agg) 
    crit_leaf=Loss_Leaf(args.T_agg) 
    optimizer = torch.optim.SGD(node_origin.model.parameters(), lr=node_origin.args.lr, momentum=0.9, weight_decay=0.001)

    select_data = data_selection(noises,labels,args.s)
    
    for idx,(noise,label) in enumerate(zip(noises,labels)): 
        if idx in select_data:
            optimizer.zero_grad()
            fake_data=node_neigh.autoencoder.decoder(noise) 
            nei_logits,_=node_neigh.model(fake_data) 
            logits_fake,_=node_origin.model(fake_data) 
            loss=0.0
            if node_origin.is_leaf(): 
                img,label_=node_origin.dataset[idx]
                img,label_=img.cuda(),label_.cuda()
                assert torch.equal(label, label_)
                logits_true,_=node_origin.model(img) 
                loss=loss+crit_leaf(logits_fake,nei_logits,logits_true,label.long()) 
            else: 
                loss=loss+crit_non_leaf(logits_fake,nei_logits,label.long())
            loss.backward(retain_graph=True) 
            optimizer.step()  
    

    update_teacher(node_origin,node_neigh,noises,labels,args)