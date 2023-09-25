import torch
import numpy as np

def computeCentersLengthsNoFaces(v):
    centers = (v[0:-1,:] + v[1:,:])/2.0
    lenel = v[1:,:] - v[0:-1,:]
    return (centers,lenel)

# face centers and area weighted normal
def computeCentersAreas(f,v):
    xDef1 = v[f[:, 0].clone(), :].clone()
    xDef2 = v[f[:, 1].clone(), :].clone()
    xDef3 = v[f[:, 2].clone(), :].clone()
    centers = (xDef1 + xDef2 + xDef3) / 3
    surfel =  torch.cross(xDef2-xDef1, xDef3-xDef1)
    return (centers,surfel)

def currentNorm(centers1,lenel1,centers2,lenel2,sigma):
    g11 = kernelMatrixGauss(centers1,sigma)
    obj0 = torch.sum(torch.mm(torch.reshape(lenel1,(lenel1.shape[0],lenel1.shape[1])),torch.reshape(lenel1,(lenel1.shape[1],lenel1.shape[0]))) * g11)
    
    g22 = kernelMatrixGauss(centers2,sigma)
    g12 = kernelMatrixGauss2(centers1,centers2,sigma)
    obj = obj0 + torch.sum(torch.mm(torch.reshape(lenel2,(lenel2.shape[0],lenel2.shape[1])),torch.reshape(lenel2,(lenel2.shape[1],lenel2.shape[0]))) * g22) - 2*torch.sum(torch.mm(torch.reshape(lenel2,(lenel2.shape[0],lenel2.shape[1])),torch.reshape(lenel1,(lenel1.shape[1],lenel1.shape[0]))) * g12)
    return obj

def kernelMatrixGauss(centers,sigma):
    self_distmat = torch.sqrt((torch.reshape(centers[:,0],(centers.shape[0],1)).repeat((1,centers.shape[0])) - torch.reshape(centers[:,0],(1,centers.shape[0])).repeat((centers.shape[0],1)))**2 + (torch.reshape(centers[:,1],(centers.shape[0],1)).repeat((1,centers.shape[0])) - torch.reshape(centers[:,1],(1,centers.shape[0])).repeat((centers.shape[0],1)))**2 + (torch.reshape(centers[:,2],(centers.shape[0],1)).repeat((1,centers.shape[0])) - torch.reshape(centers[:,2],(1,centers.shape[0])).repeat((centers.shape[0],1)))**2 )
    #self_distmat = tf.where(self_distmat < 1e-5, 1e-5*tf.ones_like(self_distmat), self_distmat)
    K = torch.exp(-1.0*self_distmat**2 / (2*sigma**2))
    return K

def kernelMatrixGauss2(centers1,centers2,sigma):
    pointdistmat = torch.sqrt((torch.reshape(centers1[:,0],(centers1.shape[0],1)).repeat((1,centers2.shape[0])) - torch.reshape(centers2[:,0],(1,centers2.shape[0])).repeat((centers1.shape[0],1)))**2 + (torch.reshape(centers1[:,1],(centers1.shape[0],1)).repeat((1,centers2.shape[0])) - torch.reshape(centers2[:,1],(1,centers2.shape[0])).repeat((centers1.shape[0],1)))**2  + (torch.reshape(centers1[:,2],(centers1.shape[0],1)).repeat((1,centers2.shape[0])) - torch.reshape(centers2[:,2],(1,centers2.shape[0])).repeat((centers1.shape[0],1)))**2)
    #pointdistmat = tf.where(pointdistmat < 1e-5, 1e-5*tf.ones_like(pointdistmat), pointdistmat)
    K = torch.exp(-1.0*pointdistmat**2 / (2*sigma**2))
    return K

def kernelApplyK(centers,surfel,sigma):
    self_distmat = (torch.reshape(centers[:,0],(centers.shape[0],1)).repeat((1,centers.shape[0])) - torch.reshape(centers[:,0],(1,centers.shape[0])).repeat((centers.shape[0],1)))**2 + (torch.reshape(centers[:,1],(centers.shape[0],1)).repeat((1,centers.shape[0])) - torch.reshape(centers[:,1],(1,centers.shape[0])).repeat((centers.shape[0],1)))**2 + (torch.reshape(centers[:,2],(centers.shape[0],1)).repeat((1,centers.shape[0])) - torch.reshape(centers[:,2],(1,centers.shape[0])).repeat((centers.shape[0],1)))**2 
    ut = torch.exp(-0.5*(self_distmat)/(sigma**2))
    K = torch.transpose(torch.stack([torch.sum(ut*torch.reshape(surfel[:,0],(1,surfel.shape[0])).repeat(surfel.shape[0],1),dim=1),torch.sum(ut*torch.reshape(surfel[:,1],(1,surfel.shape[0])).repeat(surfel.shape[0],1),dim=1),torch.sum(ut*torch.reshape(surfel[:,2],(1,surfel.shape[0])).repeat(surfel.shape[0],1),dim=1)],dim=0),0,1)
    return K

def kernelApplyK2(centers2,surfel2,centers1,sigma):
    pointdistmat = (torch.reshape(centers1[:,0],(centers1.shape[0],1)).repeat((1,centers2.shape[0])) - torch.reshape(centers2[:,0],(1,centers2.shape[0])).repeat((centers1.shape[0],1)))**2 + (torch.reshape(centers1[:,1],(centers1.shape[0],1)).repeat((1,centers2.shape[0])) - torch.reshape(centers2[:,1],(1,centers2.shape[0])).repeat((centers1.shape[0],1)))**2  + (torch.reshape(centers1[:,2],(centers1.shape[0],1)).repeat((1,centers2.shape[0])) - torch.reshape(centers2[:,2],(1,centers2.shape[0])).repeat((centers1.shape[0],1)))**2
    ut = torch.exp(-0.5*(pointdistmat)/(sigma**2))
    K = torch.transpose(torch.stack([torch.sum(ut*torch.reshape(surfel2[:,0],(1,surfel2.shape[0])).repeat(centers1.shape[0],1),dim=1),torch.sum(ut*torch.reshape(surfel2[:,1],(1,surfel2.shape[0])).repeat(centers1.shape[0],1),dim=1),torch.sum(ut*torch.reshape(surfel2[:,2],(1,surfel2.shape[0])).repeat(centers1.shape[0],1),dim=1)],dim=0),0,1)
    return K


def currentNorm0(centers,surfel,sigma):
    val = torch.sum(surfel * kernelApplyK(centers,surfel,sigma))
    return val


def currentNormDef(centers1,surfel1,centers2,surfel2,sigma):
    val = torch.sum(surfel1*kernelApplyK(centers1,surfel1,sigma)) - 2* torch.sum(surfel1*kernelApplyK2(centers2,surfel2,centers1,sigma))
    return val

# here, 2 is target, 1 is deforming template
def currentNormSum(centers1,surfel1,centers2,surfel2,sigma):
    return currentNorm0(centers2,surfel2,sigma) + currentNormDef(centers1,surfel1,centers2,surfel2,sigma)
