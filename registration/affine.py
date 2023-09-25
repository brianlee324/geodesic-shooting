import torch
import numpy as np

# rigid alignment
def rigid(q,R,R_center,translation,device='cpu'):
    # apply rigid motion
    if len(q.shape) == 1:
        q = q - R_center
        q = torch.reshape(torch.mm(R,torch.reshape(q,(3,1))) + torch.reshape(translation[:,0],(3,1)),(1,3))
        q = q + R_center
    else:
        q = q - R_center
        q = torch.transpose(torch.mm(R,torch.transpose(q,0,1)) + torch.reshape(translation[:,0],(3,1)),0,1)
        q = q + R_center

    return q