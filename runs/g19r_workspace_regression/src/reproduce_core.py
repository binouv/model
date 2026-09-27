import torch
import torch.nn as nn
import torch.nn.functional as F

PAD,BOS,EOS=0,1,2
VOCAB=259

def benc(s): return [x+3 for x in s.encode("utf-8")]

def norm_state(v): return (2.0*(v+128)/383.0)-1.0

class GatedDeltaLM(nn.Module):
    def __init__(self, hidden=160, layers=2):
        super().__init__()
        self.emb=nn.Embedding(VOCAB,hidden,padding_idx=PAD)
        self.gru=nn.GRU(hidden,hidden,num_layers=layers,batch_first=True)
        self.mix=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden))
        self.gate=nn.Linear(hidden,hidden)
        self.norm=nn.LayerNorm(hidden)
        self.lm=nn.Linear(hidden,VOCAB)
        self.ws=nn.Linear(hidden,1)
    def forward(self,x,h=None):
        y,h2=self.gru(self.emb(x),h)
        z=self.norm(y+torch.sigmoid(self.gate(y))*self.mix(y))
        return self.lm(z),self.ws(z).squeeze(-1),h2,z

def paired_loss(model,x,y,answer_mask,marker_pos,states,arm):
    logits,ws,_,_=model(x)
    answer_ce=F.cross_entropy(logits[answer_mask],y[answer_mask])
    idx=torch.arange(x.size(0)).unsqueeze(1)
    pred=ws[idx,marker_pos]
    target=states.float()
    if arm=="bag_state_regression":
        target,_=torch.sort(target,dim=1)
    target=torch.tensor([[norm_state(float(v)) for v in row] for row in target])
    aux=F.smooth_l1_loss(pred,target,beta=0.1)
    return answer_ce + (0.0 if arm=="final_only_control" else 0.20*aux), answer_ce, aux

# Exact executed script, generated data, raw endpoint metrics and checkpoint bytes are
# preserved in the local downloadable G19R bundle. This compact file preserves the
# architecture and paired objective contract for repository review.
