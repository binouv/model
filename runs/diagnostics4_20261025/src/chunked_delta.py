"""Optional bounded-chunk implementation of the existing FlyGraph recurrence.

No imported third-party implementation, new parameters, or architecture change.
The original exact-math recurrence is applied to chunks, carrying its FP32 state.
Floating-point reassociation means close equivalence, not bitwise identical resume.
Global GQA layers retain their original quadratic prefill cost.
"""
from model import delta_rule as original_delta_rule
import torch

def chunked_delta_rule(q,k,v,log_decay,beta,state=None,chunk_size=64):
    if chunk_size<1:raise ValueError('chunk_size must be positive')
    out=[]
    for start in range(0,q.shape[2],chunk_size):
        stop=min(start+chunk_size,q.shape[2])
        y,state=original_delta_rule(q[:,:,start:stop],k[:,:,start:stop],v[:,:,start:stop],log_decay[:,:,start:stop],beta[:,:,start:stop],state)
        out.append(y)
    return torch.cat(out,dim=2),state
