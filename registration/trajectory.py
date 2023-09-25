import torch
import numpy as np

# definition of 3d gaussian is 1/(sigma^3*(2*pi)^(3/2)) * exp( -0.5/(sigma^2)*((x[0] - mu[0])^2 + (x[1] - mu[1])^2 + (x[2] - mu[2])^2))
def gauss(x,mu,sigma):
    if len(mu.shape) == 1:
        return 1/(sigma**3 * (2*np.pi)**(3/2)) * torch.exp( -0.5/(sigma**2) * ((x[0] - mu[0])**2 + (x[1] - mu[1])**2 + (x[2] - mu[2])**2))
    else:
        return 1/(sigma**3 * (2*np.pi)**(3/2)) * torch.exp( -0.5/(sigma**2) * torch.sum((x-mu)**2,dim=1))

# definition of derivative of 3d gaussian is 1/((2*pi)^(3/2) * sigma^5) * (mu-x) * exp( -0.5/(sigma^2)*((x[0] - mu[0])^2 + (x[1] - mu[1])^2 + (x[2] - mu[2])^2))
def d_gauss(x,mu,sigma):
    if len(mu.shape) == 1:
        return 1/(sigma**5 * (2*np.pi)**(3/2)) * (mu-x) * torch.exp( -0.5/(sigma**2) * ((x[0] - mu[0])**2 + (x[1] - mu[1])**2 + (x[2] - mu[2])**2))
    else:
        return 1/(sigma**5 * (2*np.pi)**(3/2)) * (mu-x) * torch.reshape(torch.exp( -0.5/(sigma**2) * torch.sum((x-mu)**2,dim=1)),(mu.shape[0],1))

# definition of 3d gaussian is 1/(sigma^3*(2*pi)^(3/2)) * exp( -0.5/(sigma^2)*((x[0] - mu[0])^2 + (x[1] - mu[1])^2 + (x[2] - mu[2])^2))
def gaussVectorized(x,mu,sigma):
    # output is mu.shape[0] x x.shape[0]
    if len(mu.shape) == 1:
        return 1/(sigma**3 * (2*np.pi)**(3/2)) * torch.exp( -0.5/(sigma**2) * ((x[0] - mu[0])**2 + (x[1] - mu[1])**2 + (x[2] - mu[2])**2))
    else:
        return 1/(sigma**3 * (2*np.pi)**(3/2)) * torch.exp( -0.5/(sigma**2) * torch.sum((torch.transpose(x.unsqueeze(2),0,2)-mu.unsqueeze(2).repeat(1,1,x.shape[0]))**2,dim=1))

# definition of derivative of 3d gaussian is 1/((2*pi)^(3/2) * sigma^5) * (mu-x) * exp( -0.5/(sigma^2)*((x[0] - mu[0])^2 + (x[1] - mu[1])^2 + (x[2] - mu[2])^2))

def d_gaussVectorized(x,mu,sigma):
    # output is mu.shape[0] x 3 x x.shape[0]
    if len(mu.shape) == 1:
        return 1/(sigma**5 * (2*np.pi)**(3/2)) * (mu-x) * torch.exp( -0.5/(sigma**2) * ((x[0] - mu[0])**2 + (x[1] - mu[1])**2 + (x[2] - mu[2])**2))
    else:
        return 1/(sigma**5 * (2*np.pi)**(3/2)) * (mu.unsqueeze(2).repeat((1,1,x.shape[0]))-torch.transpose(x.unsqueeze(2),0,2)) * torch.exp( -0.5/(sigma**2) * torch.sum((torch.transpose(x.unsqueeze(2),0,2)-mu.unsqueeze(2).repeat((1,1,x.shape[0])))**2,dim=1)).unsqueeze(1)


# definition of 3d gaussian is 1/(sigma^3*(2*pi)^(3/2)) * exp( -0.5/(sigma^2)*((x[0] - mu[0])^2 + (x[1] - mu[1])^2 + (x[2] - mu[2])^2))
def gaussVectorizedNoNorm(x,mu,sigma):
    # output is mu.shape[0] x x.shape[0]
    if len(mu.shape) == 1:
        return torch.exp( -0.5/(sigma**2) * ((x[0] - mu[0])**2 + (x[1] - mu[1])**2 + (x[2] - mu[2])**2))
    else:
        return torch.exp( -0.5/(sigma**2) * torch.sum((torch.transpose(x.unsqueeze(2),0,2)-mu.unsqueeze(2).repeat(1,1,x.shape[0]))**2,dim=1))

# definition of derivative of 3d gaussian is 1/((2*pi)^(3/2) * sigma^5) * (mu-x) * exp( -0.5/(sigma^2)*((x[0] - mu[0])^2 + (x[1] - mu[1])^2 + (x[2] - mu[2])^2))

def d_gaussVectorizedNoNorm(x,mu,sigma):
    # output is mu.shape[0] x 3 x x.shape[0]
    if len(mu.shape) == 1:
        return 1/sigma**2 * (mu-x) * torch.exp( -0.5/(sigma**2) * ((x[0] - mu[0])**2 + (x[1] - mu[1])**2 + (x[2] - mu[2])**2))
    else:
        return 1/sigma**2 * (mu.unsqueeze(2).repeat((1,1,x.shape[0]))-torch.transpose(x.unsqueeze(2),0,2)) * torch.exp( -0.5/(sigma**2) * torch.sum((torch.transpose(x.unsqueeze(2),0,2)-mu.unsqueeze(2).repeat((1,1,x.shape[0])))**2,dim=1)).unsqueeze(1)



def computeU(q,p0,grid,nT,sigma):
    # here grid should be nx3, so it will need to be reshaped
    p=p0.clone()
    u_grid = []
    
    if len(q.shape) == 1:
        npoints = 1
    else:
        npoints = q.shape[0]
        
    grid_points = grid.shape[0]
    
    for t in range(1,len(nT)):
        u = torch.transpose(torch.sum(p.clone().unsqueeze(2).repeat((1,1,npoints)) * gaussVectorizedNoNorm(q,q,sigma).unsqueeze(1), dim=0),0,1)
        #u_grid.append(torch.transpose(torch.sum(p.clone().unsqueeze(2).repeat((1,1,grid_points)) * gaussVectorizedNoNorm(grid,q,sigma).unsqueeze(1), dim=0),0,1).type(torch.FloatTensor))
        u_grid.append(torch.transpose(torch.sum(p.clone().unsqueeze(2).repeat((1,1,grid_points)) * gaussVectorizedNoNorm(grid,q,sigma).unsqueeze(1), dim=0),0,1))
        p_dot = torch.squeeze(torch.matmul(-torch.matmul(torch.transpose(p.clone(),0,1),d_gaussVectorizedNoNorm(q,q,sigma).permute((2,0,1))), p.clone().unsqueeze(2)))

        # update p and q
        p = p + p_dot.clone()*((nT[t]-nT[t-1]))
        q = q + u.clone()*((nT[t]-nT[t-1]))
        
    return u_grid
                     
def reg(q,p,sigma):
    regsum = 0
    for i in range(q.shape[0]):
        for ii in range(q.shape[0]):
            regsum += torch.dot(p[i,:],p[ii,:]) * gauss(q[i,:],q[ii,:],sigma)
    
    return regsum*0.5

def regVectorized(q,p,sigma):
    regsum = 0
    for i in range(q.shape[0]):
        regsum += torch.sum(torch.sum(p*p[i,:].repeat((p.shape[0],1)),dim=1) * gauss(q[i,:],q,sigma))
    
    return regsum*0.5

def regVectorizedPlus(q,p,sigma):
    regsum = 0
    regsum = torch.sum(torch.sum(p.unsqueeze(1).repeat(1,p.shape[0],1)*p.unsqueeze(0).repeat(p.shape[0],1,1),dim=2)*gaussVectorized(q,q,sigma))
    
    return regsum*0.5


def regVectorizedPlusNoNorm(q,p,sigma):
    regsum = 0
    regsum = torch.sum(torch.sum(p.unsqueeze(1).repeat(1,p.shape[0],1)*p.unsqueeze(0).repeat(p.shape[0],1,1),dim=2)*gaussVectorizedNoNorm(q,q,sigma))
    
    return regsum*0.5
