import torch
import numpy as np
import sys
sys.path.insert(0,"../")
from cost_functions.currentMatching import *
from registration.affine import *
from registration.nonlinear import *
from registration.trajectory import *

# rigid registration of surfaces
def register_rigid(faces_template,vertices_template,faces_target,vertices_target,niter=200,sigma=12,reg_weight=35000,device="cpu"):
    # center the centroids
    centroid_template = np.mean(vertices_template,axis=0)
    centroid_target = np.mean(vertices_target,axis=0)
    vertices_template -= [centroid_target]
    
    # initialize parameters
    theta_x = (torch.zeros((1))).to(device=device).requires_grad_(True)
    theta_y = (torch.zeros((1))).to(device=device).requires_grad_(True)
    theta_z = (torch.zeros((1))).to(device=device).requires_grad_(True)
    translation = (torch.rand(3,1)*0.0001).to(device=device).requires_grad_(True)
    
    # specify center of rotation
    R_center = torch.mean(torch.Tensor(vertices_template.copy()),dim=0).type(torch.FloatTensor).to(device=device)
    
    # calculate surface info for currents
    centers_target,surfel_target = computeCentersAreas(torch.Tensor(faces_target.copy()).type(torch.LongTensor),torch.Tensor(vertices_target.copy()))
    centers_target = centers_target.to(device=device)
    surfel_target = surfel_target.to(device=device)
    
    # specify optimizer
    optimizer = torch.optim.Adam([theta_x,theta_y,theta_z,translation], lr=0.2)
    
    # main loop
    for i in range(niter):
        # produce rotation matrix
        # this is a hack to preserve gradient
        Rx = torch.cos(theta_x) * torch.tensor([[0,0,0],[0,1,0],[0,0,0]]).type(torch.FloatTensor).to(device=device) + torch.cos(theta_x) * torch.tensor([[0,0,0],[0,0,0],[0,0,1]]).type(torch.FloatTensor).to(device=device) - torch.sin(theta_x) * torch.tensor([[0,0,0],[0,0,1],[0,0,0]]).type(torch.FloatTensor).to(device=device) + torch.sin(theta_x) * torch.tensor([[0,0,0],[0,0,0],[0,1,0]]).type(torch.FloatTensor).to(device=device) + torch.tensor([[1,0,0],[0,0,0],[0,0,0]]).type(torch.FloatTensor).to(device=device)
        Ry = torch.cos(theta_y) * torch.tensor([[1,0,0],[0,0,0],[0,0,1]]).type(torch.FloatTensor).to(device=device) + torch.sin(theta_y) * torch.tensor([[0,0,1],[0,0,0],[-1,0,0]]).type(torch.FloatTensor).to(device=device) + torch.tensor([[0,0,0],[0,1,0],[0,0,0]]).type(torch.FloatTensor).to(device=device)
        Rz = torch.cos(theta_z) * torch.tensor([[1,0,0],[0,1,0],[0,0,0]]).type(torch.FloatTensor).to(device=device) + torch.sin(theta_z) * torch.tensor([[0,-1,0],[1,0,0],[0,0,0]]).type(torch.FloatTensor).to(device=device) + torch.tensor([[0,0,0],[0,0,0],[0,0,1]]).type(torch.FloatTensor).to(device=device)
        R = torch.mm(torch.mm(Rx,Ry),Rz)
        
        # rigid registration
        q = rigid(torch.Tensor(vertices_template).type(torch.FloatTensor).to(device=device),R,R_center,translation,device=device)
        
        # currents matching
        centers_template,surfel_template = computeCentersAreas(torch.Tensor(faces_template.copy()).type(torch.LongTensor),q)
        matchloss = currentNormSum(centers_template,surfel_template,centers_target,surfel_target,sigma)
        
        # backprop
        matchloss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    return q.detach().cpu().numpy(), [x.detach().cpu().numpy() for x in [theta_x,theta_y,theta_z,translation]]

# affine registration of surfaces
def register_affine(faces_template,vertices_template,faces_target,vertices_target,niter=200,sigma=12,reg_weight=35000,device="cpu"):
    # center the centroids
    centroid_template = np.mean(vertices_template,axis=0)
    centroid_target = np.mean(vertices_target,axis=0)
    vertices_template -= [centroid_target]
    
    # initialize parameters
    theta_x = (torch.zeros((1))).to(device=device).requires_grad_(True)
    theta_y = (torch.zeros((1))).to(device=device).requires_grad_(True)
    theta_z = (torch.zeros((1))).to(device=device).requires_grad_(True)
    translation = (torch.rand(3,1)*0.0001).to(device=device).requires_grad_(True)
    scale_x = torch.ones((1)).to(device=device).requires_grad_(True)
    scale_y = torch.ones((1)).to(device=device).requires_grad_(True)
    scale_z = torch.ones((1)).to(device=device).requires_grad_(True)
    shear_x = torch.zeros((1)).to(device=device).requires_grad_(True)
    shear_y = torch.zeros((1)).to(device=device).requires_grad_(True)
    shear_z = torch.zeros((1)).to(device=device).requires_grad_(True)
    
    # specify center of rotation
    R_center = torch.mean(torch.Tensor(vertices_template.copy()),dim=0).type(torch.FloatTensor).to(device=device)
    
    # calculate surface info for currents
    centers_target,surfel_target = computeCentersAreas(torch.Tensor(faces_target.copy()).type(torch.LongTensor),torch.Tensor(vertices_target.copy()))
    centers_target = centers_target.to(device=device)
    surfel_target = surfel_target.to(device=device)
    
    # specify optimizer
    optimizer = torch.optim.Adam([theta_x,theta_y,theta_z,translation,shear_x,shear_y,shear_z,scale_x,scale_y,scale_z], lr=0.2)
    
    for i in range(niter):
        Rx = torch.cos(theta_x) * torch.tensor([[0,0,0],[0,1,0],[0,0,0]]).type(torch.FloatTensor).to(device=device) + torch.cos(theta_x) * torch.tensor([[0,0,0],[0,0,0],[0,0,1]]).type(torch.FloatTensor).to(device=device) - torch.sin(theta_x) * torch.tensor([[0,0,0],[0,0,1],[0,0,0]]).type(torch.FloatTensor).to(device=device) + torch.sin(theta_x) * torch.tensor([[0,0,0],[0,0,0],[0,1,0]]).type(torch.FloatTensor).to(device=device) + torch.tensor([[1,0,0],[0,0,0],[0,0,0]]).type(torch.FloatTensor).to(device=device)
        Ry = torch.cos(theta_y) * torch.tensor([[1,0,0],[0,0,0],[0,0,1]]).type(torch.FloatTensor).to(device=device) + torch.sin(theta_y) * torch.tensor([[0,0,1],[0,0,0],[-1,0,0]]).type(torch.FloatTensor).to(device=device) + torch.tensor([[0,0,0],[0,1,0],[0,0,0]]).type(torch.FloatTensor).to(device=device)
        Rz = torch.cos(theta_z) * torch.tensor([[1,0,0],[0,1,0],[0,0,0]]).type(torch.FloatTensor).to(device=device) + torch.sin(theta_z) * torch.tensor([[0,-1,0],[1,0,0],[0,0,0]]).type(torch.FloatTensor).to(device=device) + torch.tensor([[0,0,0],[0,0,0],[0,0,1]]).type(torch.FloatTensor).to(device=device)
        R = torch.mm(torch.mm(Rx,Ry),Rz)

        scale_mat = scale_x * torch.tensor([[1,0,0],[0,0,0],[0,0,0]]).type(torch.FloatTensor).to(device=device) + scale_y * torch.tensor([[0,0,0],[0,1,0],[0,0,0]]).type(torch.FloatTensor).to(device=device) + scale_z * torch.tensor([[0,0,0],[0,0,0],[0,0,1]]).type(torch.FloatTensor).to(device=device)

        shear_mat_x = shear_y * torch.tensor([[0,0,0],[1,0,0],[0,0,0]]).type(torch.FloatTensor).to(device=device) + shear_z * torch.tensor([[0,0,0],[0,0,0],[1,0,0]]).type(torch.FloatTensor).to(device=device) + torch.eye(3).to(device=device)
        shear_mat_y = shear_x * torch.tensor([[0,1,0],[0,0,0],[0,0,0]]).type(torch.FloatTensor).to(device=device) + shear_z * torch.tensor([[0,0,0],[0,0,0],[0,1,0]]).type(torch.FloatTensor).to(device=device) + torch.eye(3).to(device=device)
        shear_mat_z = shear_x * torch.tensor([[0,0,1],[0,0,0],[0,0,0]]).type(torch.FloatTensor).to(device=device) + shear_y * torch.tensor([[0,0,0],[0,0,1],[0,0,0]]).type(torch.FloatTensor).to(device=device) + torch.eye(3).to(device=device)
        shear_mat = torch.mm(torch.mm(shear_mat_x,shear_mat_y), shear_mat_z)

        A = torch.mm(torch.mm(R, shear_mat), scale_mat)

        q = rigid(torch.Tensor(vertices_template).type(torch.FloatTensor).to(device=device),A,R_center,translation,device='cpu')

        centers_template,surfel_template = computeCentersAreas(torch.Tensor(faces_template.copy()).type(torch.LongTensor),q)

        matchloss = currentNormSum(centers_template,surfel_template,centers_target,surfel_target,sigma) 

        #print('iter '+ str(i) + ': E= ' + str(matchloss.item()))
        # take a backprop step
        matchloss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return q.detach().cpu().numpy(), [x.detach().cpu().numpy() for x in [theta_x, theta_y, theta_z, translation, scale_x, scale_y, scale_z, shear_x, shear_y, shear_z]]
    

# nonlinear registration of surfaces
def register_nonlinear(faces_template,vertices_template,faces_target,vertices_target,niter=200,nT=[0,1],sigma=12,reg_weight=35000,device="cpu"):
    # center the centroids
    centroid_template = np.mean(vertices_template,axis=0)
    centroid_target = np.mean(vertices_target,axis=0)
    vertices_template -= [centroid_target]
    
    # initialize parameters
    theta_x = (torch.zeros((1))).to(device=device).requires_grad_(True)
    theta_y = (torch.zeros((1))).to(device=device).requires_grad_(True)
    theta_z = (torch.zeros((1))).to(device=device).requires_grad_(True)
    translation = (torch.rand(3,1)*0.0001).to(device=device).requires_grad_(True)
    scale_x = torch.ones((1)).to(device=device).requires_grad_(True)
    scale_y = torch.ones((1)).to(device=device).requires_grad_(True)
    scale_z = torch.ones((1)).to(device=device).requires_grad_(True)
    shear_x = torch.zeros((1)).to(device=device).requires_grad_(True)
    shear_y = torch.zeros((1)).to(device=device).requires_grad_(True)
    shear_z = torch.zeros((1)).to(device=device).requires_grad_(True)
    p_0 = (((torch.rand((vertices_template.shape[0],3)).type(torch.FloatTensor)-0.5)) * 0.0).to(device=device).requires_grad_(True)
    q_0 = torch.Tensor(vertices_template.copy()).type(torch.FloatTensor).to(device=device)
    
    # specify center of rotation
    R_center = torch.mean(torch.Tensor(vertices_template.copy()),dim=0).type(torch.FloatTensor).to(device=device)
    
    # calculate surface info for currents
    centers_target,surfel_target = computeCentersAreas(torch.Tensor(faces_target.copy()).type(torch.LongTensor),torch.Tensor(vertices_target.copy()))
    centers_target = centers_target.to(device=device)
    surfel_target = surfel_target.to(device=device)
    
    # specify optimizer    
    optimizer = torch.optim.Adam([{"params": (theta_x,theta_y,theta_z,translation,scale_x,scale_y,scale_z,shear_x,shear_y,shear_z)},{"params":p_0,"lr": 0.001}], lr=0.02)
    
    for i in range(niter):
        Rx = torch.cos(theta_x) * torch.tensor([[0,0,0],[0,1,0],[0,0,0]]).type(torch.FloatTensor).to(device=device) + torch.cos(theta_x) * torch.tensor([[0,0,0],[0,0,0],[0,0,1]]).type(torch.FloatTensor).to(device=device) - torch.sin(theta_x) * torch.tensor([[0,0,0],[0,0,1],[0,0,0]]).type(torch.FloatTensor).to(device=device) + torch.sin(theta_x) * torch.tensor([[0,0,0],[0,0,0],[0,1,0]]).type(torch.FloatTensor).to(device=device) + torch.tensor([[1,0,0],[0,0,0],[0,0,0]]).type(torch.FloatTensor).to(device=device)
        Ry = torch.cos(theta_y) * torch.tensor([[1,0,0],[0,0,0],[0,0,1]]).type(torch.FloatTensor).to(device=device) + torch.sin(theta_y) * torch.tensor([[0,0,1],[0,0,0],[-1,0,0]]).type(torch.FloatTensor).to(device=device) + torch.tensor([[0,0,0],[0,1,0],[0,0,0]]).type(torch.FloatTensor).to(device=device)
        Rz = torch.cos(theta_z) * torch.tensor([[1,0,0],[0,1,0],[0,0,0]]).type(torch.FloatTensor).to(device=device) + torch.sin(theta_z) * torch.tensor([[0,-1,0],[1,0,0],[0,0,0]]).type(torch.FloatTensor).to(device=device) + torch.tensor([[0,0,0],[0,0,0],[0,0,1]]).type(torch.FloatTensor).to(device=device)
        R = torch.mm(torch.mm(Rx,Ry),Rz)

        scale_mat = scale_x * torch.tensor([[1,0,0],[0,0,0],[0,0,0]]).type(torch.FloatTensor).to(device=device) + scale_y * torch.tensor([[0,0,0],[0,1,0],[0,0,0]]).type(torch.FloatTensor).to(device=device) + scale_z * torch.tensor([[0,0,0],[0,0,0],[0,0,1]]).type(torch.FloatTensor).to(device=device)

        shear_mat_x = shear_y * torch.tensor([[0,0,0],[1,0,0],[0,0,0]]).type(torch.FloatTensor).to(device=device) + shear_z * torch.tensor([[0,0,0],[0,0,0],[1,0,0]]).type(torch.FloatTensor).to(device=device) + torch.eye(3).to(device=device)
        shear_mat_y = shear_x * torch.tensor([[0,1,0],[0,0,0],[0,0,0]]).type(torch.FloatTensor).to(device=device) + shear_z * torch.tensor([[0,0,0],[0,0,0],[0,1,0]]).type(torch.FloatTensor).to(device=device) + torch.eye(3).to(device=device)
        shear_mat_z = shear_x * torch.tensor([[0,0,1],[0,0,0],[0,0,0]]).type(torch.FloatTensor).to(device=device) + shear_y * torch.tensor([[0,0,0],[0,0,1],[0,0,0]]).type(torch.FloatTensor).to(device=device) + torch.eye(3).to(device=device)
        shear_mat = torch.mm(torch.mm(shear_mat_x,shear_mat_y), shear_mat_z)

        A = torch.mm(torch.mm(R, shear_mat), scale_mat)
        
        translation_t = torch.zeros((3,len(nT)))
        translation_t[:,0] = torch.squeeze(translation)

        A_list = [A]
        for ii in range(len(nT)-2):
            A_list.append(torch.eye(3))

        q,p,q_save,p_save = shootUnevenVectorizedWithRigidNoGaussNorm(q_0,p_0,A_list,R_center,translation_t,nT,sigma,device=device)
        regloss = regVectorizedPlus(q_0,p_0,sigma)

        centers_template,surfel_template = computeCentersAreas(torch.Tensor(faces_template.copy()).type(torch.LongTensor),q)

        matchloss = currentNormSum(centers_template,surfel_template,centers_target,surfel_target,sigma)

        loss = matchloss + regloss*reg_weight

        #print('iter '+ str(i) + ': Em= ' + str(matchloss.item())  + ', Er= ' + str(regloss.item()))
        # take a backprop step
        matchloss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return q.detach().cpu().numpy(), [x.detach().cpu().numpy() for x in [theta_x, theta_y, theta_z, translation, scale_x, scale_y, scale_z, shear_x, shear_y, shear_z, p_0]]
   
    