import torchmetrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
cpu = torch.device('cpu')


def DM_rate(output,target,x_control):
    prot_att=x_control
    index_prot=torch.squeeze(torch.nonzero(prot_att[:] != 1.))
    target_prot=torch.index_select(target, 0, index=index_prot)
    index_prot_pos=torch.squeeze(torch.nonzero(target_prot[:] == 1. ))
    index_prot_neg=torch.squeeze(torch.nonzero(target_prot[:] == 0. ))

    index_non_prot=torch.squeeze(torch.nonzero(prot_att[:] == 1.))
    target_non_prot=torch.index_select(target, 0, index=index_non_prot)
    index_non_prot_pos=torch.squeeze(torch.nonzero(target_non_prot[:] == 1. ))
    index_non_prot_neg=torch.squeeze(torch.nonzero(target_non_prot[:] == 0. ))

    if index_prot_pos.shape==torch.Size([]) or index_prot_pos.shape==torch.Size([0])\
        or index_non_prot_pos.shape==torch.Size([]) or index_non_prot_pos.shape==torch.Size([0]):
            l_prot_pos=torch.tensor(0.0001)
            l_non_prot_pos=torch.tensor(0.0001)
    else:        
            l_prot_pos=acc(torch.index_select(output, 0, index=index_prot_pos),torch.index_select(target, 0, index=index_prot_pos))    
            l_non_prot_pos=acc(torch.index_select(output, 0, index=index_non_prot_pos),torch.index_select(target, 0, index=index_non_prot_pos))    
    
    if index_prot_neg.shape==torch.Size([]) or index_prot_neg.shape==torch.Size([0])\
        or index_non_prot_neg.shape==torch.Size([]) or index_non_prot_neg.shape==torch.Size([0]):
            l_prot_neg=torch.tensor(0.0001)
            l_non_prot_neg=torch.tensor(0.0001)
    else:        
            l_prot_neg=acc(torch.index_select(output, 0, index=index_prot_neg),torch.index_select(target, 0, index=index_prot_neg))    
            l_non_prot_neg=acc(torch.index_select(output, 0, index=index_non_prot_neg),torch.index_select(target, 0, index=index_non_prot_neg))  
            
    dl_pos=torch.abs(l_prot_pos-l_non_prot_pos)
    dl_neg=torch.abs(l_prot_neg-l_non_prot_neg)
    DM=dl_pos+dl_neg
    
    return DM, dl_pos

def fair_loss(output,target,x_control):
    prot_att=x_control
    index_prot=torch.squeeze(torch.nonzero(prot_att[:] != 1.))
    target_prot=torch.index_select(target, 0, index=index_prot)
    index_prot_pos=torch.squeeze(torch.nonzero(target_prot[:] == 1. ))
    index_prot_neg=torch.squeeze(torch.nonzero(target_prot[:] == 0. ))

    index_non_prot=torch.squeeze(torch.nonzero(prot_att[:] == 1.))
    target_non_prot=torch.index_select(target, 0, index=index_non_prot)
    index_non_prot_pos=torch.squeeze(torch.nonzero(target_non_prot[:] == 1. ))
    index_non_prot_neg=torch.squeeze(torch.nonzero(target_non_prot[:] == 0. ))

    l_prot_pos=F.cross_entropy(torch.index_select(output, 0, index=index_prot_pos),torch.index_select(target, 0, index=index_prot_pos))    
    l_non_prot_pos=F.cross_entropy(torch.index_select(output, 0, index=index_non_prot_pos),torch.index_select(target, 0, index=index_non_prot_pos))    
    l_non_prot_neg=F.cross_entropy(torch.index_select(output, 0, index=index_non_prot_neg),torch.index_select(target, 0, index=index_non_prot_neg))
    l_prot_neg=F.cross_entropy(torch.index_select(output, 0, index=index_prot_neg),torch.index_select(target, 0, index=index_prot_neg))    

    for l in [l_prot_pos,l_non_prot_pos,l_prot_neg,l_non_prot_neg]:
        if torch.isinf(l)==True:
            l=torch.zeros_like(l,requires_grad=True)
    dl_pos=torch.max(l_prot_pos,l_non_prot_pos)
    dl_neg=torch.max(l_prot_neg,l_non_prot_neg)
    L=dl_pos+dl_neg
    
    return L

def Update_model(model,grads_sh,omega,G_n,r_t,opti,paths):
    lr=0.001
    loss_gn=[(G_n[t]-torch.mean(G_n)*r_t[t]) for t in range(len(G_n))]
    for i in range(len(G_n)):
        d_l=0
        if loss_gn[i]>0:
            d_l+=(len(G_n)-1)/len(G_n)*G_n[i]
        elif loss_gn[i]<0:
            d_l-=(len(G_n)-1)/len(G_n)*G_n[i]
        for j in range(len(G_n)):
            if j!=i:
                if loss_gn[j]>0:
                    d_l-=(G_n[i]/len(G_n))
                elif loss_gn[j]<0:
                    d_l+=(G_n[i]/len(G_n))
        
        omega[i]-=lr*d_l

    
    for n,p in model.named_parameters():
        if p.data.shape[0]!=2 and p.grad==None:
            t_g,w_s=[],0
            for t in paths:
                flag,w=0,1
                if n.startswith('branches'):
                    br=n.split('.')[1]
                    if br in paths[t]:
                        flag=1
                elif n.startswith('shared'):
                    flag=1
                if flag==1:
                    t_g.append(t)
                    w_s+=omega[t]
            for i in range(len(t_g)):
                if i==0:
                    p.grad=(omega[t_g[i]]/w_s)*grads_sh[t_g[i]][n]
                else:
                    p.grad+=(omega[t_g[i]]/w_s)*grads_sh[t_g[i]][n]
                    
    opti.step() 
    total=sum(omega)
    for i in range(len(omega)):
        omega[i]=omega[i]/total
    return omega,model




import torch
import numpy as np
from CKA import CKA, CudaCKA

if torch.cuda.is_available():
    cka_dv='cuda:1'
else:
    cka_dv='cpu'
    
cka = CudaCKA(cka_dv)

def sim_mat(self,out=1,groups=[]):
    if isinstance(self,nn.DataParallel):
        self=self.module
    if out!=1:
        task_layers=list(self.named_children())[-2][1]
        k=0
        for task in task_layers:
            if task.startswith(str(out-1)):
                k+=1
        sim_mat=torch.zeros(k,k).to(cka_dv)
    else:
        task_layers=list(self.named_children())[-1][1]
        sim_mat=torch.zeros(len(task_layers),len(task_layers)).to(cka_dv)
    task_layers=task_layers.to(cka_dv)
    
    i=0
    for t1 in task_layers:
        v1=None
        if out!=1:
            if t1.startswith(str(out-1)):
                v1=task_layers[t1].weight.data.T
        else:
            v1=task_layers[t1].weight.data.T
        
        if v1!=None:
            j=0
            for t2 in task_layers:
                v2=None
                if out!=1:
                    if t2.startswith(str(out-1)):
                        v2=task_layers[t2].weight.data.T
                else:
                    v2=task_layers[t2].weight.data.T
                #print(i,j)
                if v2!=None:
                    if i==j:
                        sim_mat[i][j]=0
                        #sim_mat[j][i]=1
                    else:
                        if sim_mat[i][j]==0:
                            if len(groups)==0: 
                                sim_nets=cka.linear_CKA(v1, v2)
                                #print(sim_nets)
                                sim_mat[i][j]=sim_nets
                                sim_mat[j][i]=sim_nets
                            else:
                                flag=0
                                for g in range(len(groups)):
                                    if i in groups[g] and j in groups[g]:
                                        sim_nets=cka.linear_CKA(v1, v2)
                                        sim_mat[i][j]=sim_nets
                                        sim_mat[j][i]=sim_nets
                                        flag=1
                    j+=1
            i+=1
    return sim_mat


def find_groups(scores,method='agglomerative'):
    if method=='agglomerative':
        a,b=torch.sort(-scores)
        a=-a
        #if group==None:
        group=[]
        for j in range(len(a)):
            flag=True
            for g in range(len(group)):
                if j in group[g]:
                    flag=False
            if flag:
                    #print(flag)
                    group.append([j])
            #print(group)
            for i in range(len(a)):
                #print(a[i][1],j)
                if i!=j:
                    if b[i][1]==j and b[j][1]==i and a[i][j]!=0:
                        for g in range(len(group)):
                            if j in group[g] and i not in group[g]:
                                #print(j,i)
                                group[g].append(i)

    
    return group


import copy
def branches(net=None,group=None,out=1,parents={},branches=None,premod=False):
    if group==None or isinstance(group, list)==False or len(parents)==0:
        print('Insufficient Arguments Error')
        return net
    else:
        groups=copy.deepcopy(group)
        if isinstance(net,nn.DataParallel):
            net=net.module
        if premod==True:
            pretrain=copy.deepcopy(net.pretrain.to(cpu))
            pretrain.fc.requires_grad=True
        new_parents=copy.deepcopy(parents)
        if branches==None:
            branches=nn.ModuleDict()    
        if out!=1:
            shared_layers =copy.deepcopy(net.shared[:-3].to(cpu))
            pf=net.shared[-3].to(cpu)
        else:
            shared_layers =copy.deepcopy(net.shared[:-1].to(cpu))
            pf=net.shared[-1].to(cpu)   
            
        task_layers=nn.ModuleDict({ch: copy.deepcopy(net.tasks_out[ch].to(cpu)) for ch in net.tasks_out})
        for i in range(len(groups)):
            mod=nn.Linear(pf.in_features,pf.out_features)
            mod.weight.data=pf.weight.data.detach().clone()
            mod.weight.requires_grad=True
            mod.bias.data=pf.bias.data.detach().clone()
            mod.bias.requires_grad=True
            branches[str(out)+str(i)]=mod
        del net
        if out!=1: 
            for i in range(len(new_parents)):
                for j in range(len(groups)):
                    key=int(''.join(new_parents[i][-1])[1:])#key=int(''.join(new_parents[i])[-1])
                    if key in groups[j]:           
                        new_parents[i].append(str(out)+str(j))
                        break
        else:
            for i in range(len(new_parents)):
                for j in range(len(groups)):
                    if i in groups[j]:
                        new_parents[i].append(str(out)+str(j)) 
                        break

        if premod==True:
            return pretrain,shared_layers,branches,task_layers,new_parents
        else:
            return shared_layers,branches,task_layers,new_parents
        
        
        
def _get_params(self):
        params = {}
        for name, param in self.named_parameters():
            params[name] = param.data
        return params
def get_flat_params(self,dv):
        """Get flattened and concatenated params of the model."""
        params = _get_params(self)
        flat_params = torch.Tensor().to(dv)
        #if torch.cuda.is_available() and dv==torch.device('cuda'):
        #    flat_params = flat_params.cuda()
        for _, param in params.items():
            flat_params = torch.cat((flat_params, torch.flatten(param)))
        return flat_params
def _get_param_shapes(model):
        shapes = []
        for name, param in model.named_parameters():
            shapes.append((name, param.shape, param.numel()))
        return shapes