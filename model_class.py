import torch
import torch.nn as nn
import torch.nn.functional as F
cpu = torch.device('cpu')

class MTL(nn.Module):

    def __init__(self,d_in=50,tasks=2,shapes=[1024,1024,512,128]):
        super(MTL, self).__init__()
        self.tasks=tasks
        self.shapes=shapes
        self.shared=nn.Sequential()
        for s in range(len(self.shapes)):
            if s==0:
                self.shared.add_module('fc'+str(s+1),nn.Linear(d_in, self.shapes[s]))
            else:
                self.shared.add_module('fc'+str(s+1),nn.Linear(self.shapes[s-1], self.shapes[s]))
            if s!=len(self.shapes)-1:    
                self.shared.add_module('bn'+str(s+1),nn.BatchNorm1d( self.shapes[s]))
                self.shared.add_module('relu'+str(s+1),nn.ReLU())
        self.tasks_out=nn.ModuleDict()
        for t in range(self.tasks):
            torch.manual_seed(7)
            self.tasks_out['ch'+str(t)]=nn.Linear(self.shapes[-1],2) 
    def forward(self, x):
        x = F.relu(self.shared(x))

        t=[self.tasks_out['ch'+str(i)](x) for i in range(self.tasks)]
        
        return t
    
class STL(nn.Module):

    def __init__(self,d_in=50,shapes=[1024,1024,512,128]):
        super(STL, self).__init__()
        self.shapes=shapes
        self.shared=nn.Sequential()
        for s in range(len(self.shapes)):
            if s==0:
                self.shared.add_module('fc'+str(s+1),nn.Linear(d_in, self.shapes[s]))
            else:
                self.shared.add_module('fc'+str(s+1),nn.Linear(self.shapes[s-1], self.shapes[s]))
            if s!=len(self.shapes)-1:    
                self.shared.add_module('bn'+str(s+1),nn.BatchNorm1d( self.shapes[s]))
                self.shared.add_module('relu'+str(s+1),nn.ReLU())
        self.tasks_out=nn.Linear(self.shapes[-1],2)
        
    def forward(self, x):
        x = F.relu(self.shared(x))

        t=self.tasks_out(x)
        
        return t    

class MTL_vission(nn.Module):

    def __init__(self,pretrain,tasks=2,shapes=[1024,1024,512,128]):
        super(MTL_vission, self).__init__()
        self.pretrain=pretrain
        self.tasks=tasks
        self.shapes=shapes
        self.shared=nn.Sequential()
        in_fc=self.pretrain.fc.in_features
        self.pretrain.fc=nn.Linear(in_fc,self.shapes[0])
        self.bn=nn.BatchNorm1d( self.shapes[0])
        for s in range(1,len(self.shapes)):
            self.shared.add_module('fc'+str(s),nn.Linear(self.shapes[s-1], self.shapes[s]))
            if s!=len(self.shapes)-1:    
                self.shared.add_module('bn'+str(s),nn.BatchNorm1d( self.shapes[s]))
                self.shared.add_module('relu'+str(s),nn.ReLU())
        self.tasks_out=nn.ModuleDict()
        for t in range(self.tasks):
            torch.manual_seed(7)
            self.tasks_out['ch'+str(t)]=nn.Linear(self.shapes[-1],2) 
    def forward(self, x):
        x=F.relu(self.bn(self.pretrain(x)))
        x = F.relu(self.shared(x))

        t=[self.tasks_out['ch'+str(i)](x) for i in range(self.tasks)]
        
        return t  
    
class STL_vission(nn.Module):

    def __init__(self,pretrain,shapes=[1024,1024,512,128]):
        super(STL_vission, self).__init__()
        self.pretrain=pretrain
        self.shapes=shapes
        self.shared=nn.Sequential()
        in_fc=self.pretrain.fc.in_features
        self.pretrain.fc=nn.Linear(in_fc,self.shapes[0])
        self.bn=nn.BatchNorm1d( self.shapes[0])
        for s in range(1,len(self.shapes)):
            self.shared.add_module('fc'+str(s),nn.Linear(self.shapes[s-1], self.shapes[s]))
            if s!=len(self.shapes)-1:    
                self.shared.add_module('bn'+str(s),nn.BatchNorm1d( self.shapes[s]))
                self.shared.add_module('relu'+str(s),nn.ReLU())
        self.tasks_out=nn.Linear(self.shapes[-1],2) 
        
    def forward(self, x):
        x=F.relu(self.bn(self.pretrain(x)))
        x = F.relu(self.shared(x))

        t=self.tasks_out(x)
        
        return t  
    
class Branched_model(nn.Module):
    
    def __init__(self,shared_all,branches,task_layers,parents,device='cuda'):
        super(Branched_model, self).__init__()
        self.shared = shared_all
        self.branches=branches
        #self.groups=groups
        self.tasks_out=task_layers
        self.parents=parents
        self.device=device

    def forward(self, x):       
        x = F.relu(self.shared(x))
        t=[]
        for task in range(len(self.parents)):
            br=None
            #print(task)
            for gr in range(len(self.parents[task])-1,-1,-1):
                if br==None:
                    br=self.branches[self.parents[task][gr]](x)
                else:
                    br=self.branches[self.parents[task][gr]](br)
                #print(self.parents[task][gr])
                if gr!=0:
                    bn=nn.BatchNorm1d(self.branches[self.parents[task][gr]].out_features).to(self.device)
                    br=bn(br)
                br=F.relu(br)
            t.append(self.tasks_out['ch'+str(task)](br))        
        return t


class Branched_model_vission(nn.Module):
    
    def __init__(self,pretrain,shared_all,branches,task_layers,parents,device=cpu):
        super(Branched_model_vission, self).__init__()
        self.pretrain=pretrain
        self.bn=nn.BatchNorm1d(self.pretrain.fc.out_features)
        self.shared = shared_all
        self.branches=branches
        #self.groups=groups
        self.tasks_out=task_layers
        self.parents=parents
        self.device=device

    def forward(self, x):  
        x=F.relu(self.bn(self.pretrain(x)))
        x = F.relu(self.shared(x))
        t=[]
        for task in range(len(self.parents)):
            br=None
            #print(task)
            for gr in range(len(self.parents[task])-1,-1,-1):
                if br==None:
                    br=self.branches[self.parents[task][gr]](x)
                else:
                    br=self.branches[self.parents[task][gr]](br)
                #print(self.parents[task][gr])
                if gr!=0:
                    bn=nn.BatchNorm1d(self.branches[self.parents[task][gr]].out_features).to(self.device)
                    br=bn(br)
                br=F.relu(br)
            t.append(self.tasks_out['ch'+str(task)](br))        
        return t
 
  


#def save_model(model,name='pums18'):
    
     