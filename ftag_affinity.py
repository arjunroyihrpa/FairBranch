import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from my_util import DM_rate, fair_loss

def affinity_loss(inp,target,criteria, losses=[],path='',Cumulate_aff={},lamda=1,prot=None):
    inp=torch.tensor(inp).to(dv)
    model=MTL(d_in=len(features),tasks=N_tasks,shapes=shapes).to(dv)
    for i in range(len(Cumulate_aff)):
        model.load_state_dict(torch.load(path))
        model.to(dv)
        temp_opt=optim.AdamW(params=model.parameters())
        out=model(inp.float())
        loss=criteria(out[i],torch.tensor(target[i]).to(dv))+lamda*fair_loss(aff_out[j],
                                                        torch.tensor(target[j]).to(dv),torch.tensor(prot).to(dv))
        loss.backward()
        temp_opt.step()
        with torch.no_grad():
            aff_out=model(inp.float())
            for j in range(len(aff_out)):
                aff_loss=criteria(aff_out[j],torch.tensor(target[j]).to(dv))+lamda*fair_loss(aff_out[j],
                                                            torch.tensor(target[j]).to(dv),torch.tensor(prot).to(dv))
                Cumulate_aff[str(j)][str(i)]+=(1-aff_loss.to(cpu).item()/losses[j])
        del temp_opt
    del model
    return Cumulate_aff