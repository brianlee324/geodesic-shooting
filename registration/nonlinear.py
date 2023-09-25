import torch
import numpy as np
from registration.trajectory import *

# evolve points over simulation time
def shoot(q,p0,nT,sigma,device='cpu'):
    #q=q0.clone()
    if len(q.shape) == 1:
        npoints = 1
    else:
        npoints = q.shape[0]

    # make an empty array for u, the vector field
    u = torch.zeros(q.shape).type(torch.FloatTensor).to(device=device)
    p_dot = torch.zeros(q.shape).type(torch.FloatTensor).to(device=device)

    # set gaussian sigma to some number
    #sigma=2.0

    # set some number of simulation time steps
    #nT = 10
    
    p=p0.clone()
    
    p_save = []
    q_save = []
    p_save.append(p.clone())
    q_save.append(q.clone())

    # loop
    for t in range(nT):
        # update u
        if len(u.shape) == 1:
            u = torch.sum(p.clone() * torch.reshape(gauss(q,q,sigma),(npoints,1)),dim=0)
        else:
            for i in range(u.shape[0]):
                u[i,:] = torch.sum(p.clone() * torch.reshape(gauss(q[i,:],q,sigma),(npoints,1)),dim=0)
        
        # compute time derivative of p
        if len(u.shape) == 1:
            p_dot = torch.reshape(torch.mm(-torch.mm(torch.reshape(p.clone(),(3,1)),torch.reshape(d_gauss(q,q,sigma),(1,3))), torch.reshape(p.clone(),(3,1))),(1,3))
        else:
            for i in range(u.shape[0]):
                #p_dot[i,:] = -torch.sum(p * d_gauss(q[i,:],q,sigma),dim=0) * p[i,:]
                p_dot[i,:] = torch.transpose(torch.mm(-torch.mm(torch.transpose(p.clone(),0,1),d_gauss(q[i,:],q,sigma)), p[i,:].clone().unsqueeze(1)),0,1)
        
        # update p and q
        p = p + p_dot.clone()*1/nT
        q = q + u.clone()*1/nT
        p_save.append(p.clone())
        q_save.append(q.clone())
        
    
    return q,p,q_save,p_save

# evolve points over simulation time
def shootUneven(q,p0,nT,sigma,device='cpu'):
    #q=q0.clone()
    if len(q.shape) == 1:
        npoints = 1
    else:
        npoints = q.shape[0]

    # make an empty array for u, the vector field
    u = torch.zeros(q.shape).type(torch.FloatTensor).to(device=device)
    p_dot = torch.zeros(q.shape).type(torch.FloatTensor).to(device=device)

    # set gaussian sigma to some number
    #sigma=2.0

    # set some number of simulation time steps
    #nT = 10
    
    p=p0.clone()
    
    p_save = []
    q_save = []
    p_save.append(p.clone())
    q_save.append(q.clone())

    # loop
    for t in range(1,len(nT)):
        # update u
        if len(u.shape) == 1:
            u = torch.sum(p.clone() * torch.reshape(gauss(q,q,sigma),(npoints,1)),dim=0)
        else:
            for i in range(u.shape[0]):
                u[i,:] = torch.sum(p.clone() * torch.reshape(gauss(q[i,:],q,sigma),(npoints,1)),dim=0)
        
        # compute time derivative of p
        if len(u.shape) == 1:
            p_dot = torch.reshape(torch.mm(-torch.mm(torch.reshape(p.clone(),(3,1)),torch.reshape(d_gauss(q,q,sigma),(1,3))), torch.reshape(p.clone(),(3,1))),(1,3))
        else:
            for i in range(u.shape[0]):
                #p_dot[i,:] = -torch.sum(p * d_gauss(q[i,:],q,sigma),dim=0) * p[i,:]
                p_dot[i,:] = torch.transpose(torch.mm(-torch.mm(torch.transpose(p.clone(),0,1),d_gauss(q[i,:],q,sigma)), p[i,:].clone().unsqueeze(1)),0,1)
        
        # update p and q
        p = p + p_dot.clone()*((nT[t]-nT[t-1])/nT[-1])
        q = q + u.clone()*((nT[t]-nT[t-1])/nT[-1])
        p_save.append(p.clone())
        q_save.append(q.clone())
        
    
    return q,p,q_save,p_save

# evolve points over simulation time
def shootUnevenVectorized(q,p0,nT,sigma,device='cpu'):
    #q=q0.clone()
    if len(q.shape) == 1:
        npoints = 1
    else:
        npoints = q.shape[0]

    # make an empty array for u, the vector field
    u = torch.zeros(q.shape).type(torch.FloatTensor).to(device=device)
    p_dot = torch.zeros(q.shape).type(torch.FloatTensor).to(device=device)

    # set gaussian sigma to some number
    #sigma=2.0

    # set some number of simulation time steps
    #nT = 10

    p=p0.clone()

    p_save = []
    q_save = []
    p_save.append(p.clone())
    q_save.append(q.clone())
    
    #gaussVq = gaussVectorized(q,q,sigma)
    #dgaussVq = d_gaussVectorized(q,q,sigma)

    # loop
    for t in range(1,len(nT)):
        # update u
        if len(u.shape) == 1:
            u = torch.sum(p.clone() * torch.reshape(gauss(q,q,sigma),(npoints,1)),dim=0)
        else:
            u = torch.transpose(torch.sum(p.clone().unsqueeze(2).repeat((1,1,npoints)) * gaussVectorized(q,q,sigma).unsqueeze(1), dim=0),0,1)
            #for i in range(u.shape[0]):
            #    u[i,:] = torch.sum(p.clone() * torch.reshape(gauss(q[i,:],q,sigma),(npoints,1)),dim=0)

        # compute time derivative of p
        if len(u.shape) == 1:
            p_dot = torch.reshape(torch.mm(-torch.mm(torch.reshape(p.clone(),(3,1)),torch.reshape(d_gauss(q,q,sigma),(1,3))), torch.reshape(p.clone(),(3,1))),(1,3))
        else:
            p_dot = torch.squeeze(torch.matmul(-torch.matmul(torch.transpose(p.clone(),0,1),d_gaussVectorized(q,q,sigma).permute((2,0,1))), p.clone().unsqueeze(2)))
            #for i in range(u.shape[0]):
            #    #p_dot[i,:] = -torch.sum(p * d_gauss(q[i,:],q,sigma),dim=0) * p[i,:]
            #    p_dot[i,:] = torch.transpose(torch.mm(-torch.mm(torch.transpose(p.clone(),0,1),d_gauss(q[i,:],q,sigma)), p[i,:].clone().unsqueeze(1)),0,1)

        # update p and q
        p = p + p_dot.clone()*((nT[t]-nT[t-1]))
        q = q + u.clone()*((nT[t]-nT[t-1]))
        p_save.append(p.clone())
        q_save.append(q.clone())


    return q,p,q_save,p_save

# evolve points over simulation time
# nT is in units of years
def shootUnevenVectorizedWithRigid(q,p0,R,R_center,translation,nT,sigma,device='cpu'):
    #q=q0.clone()
    if len(q.shape) == 1:
        npoints = 1
    else:
        npoints = q.shape[0]

    # make an empty array for u, the vector field
    u = torch.zeros(q.shape).type(torch.FloatTensor).to(device=device)
    p_dot = torch.zeros(q.shape).type(torch.FloatTensor).to(device=device)

    # set gaussian sigma to some number
    #sigma=2.0

    # set some number of simulation time steps
    #nT = 10

    p=p0.clone()

    p_save = []
    q_save = []
    p_save.append(p.clone())
    q_save.append(q.clone())
    
    #gaussVq = gaussVectorized(q,q,sigma)
    #dgaussVq = d_gaussVectorized(q,q,sigma)

    # loop
    for t in range(1,len(nT)):
        # apply rigid motion
        if len(u.shape) == 1:
            q = q - R_center
            q = torch.reshape(torch.mm(R[t-1],torch.reshape(q,(3,1))) + torch.reshape(translation[:,t-1],(3,1)),(1,3))
            q = q + R_center
            p = torch.reshape(torch.mm(R[t-1],torch.reshape(p,(3,1))),(1,3))
        else:
            q = q - R_center
            q = torch.transpose(torch.mm(R[t-1],torch.transpose(q,0,1)) + torch.reshape(translation[:,t-1],(3,1)),0,1)
            q = q + R_center
            p = torch.transpose(torch.mm(R[t-1],torch.transpose(p,0,1)),0,1)
                              
        # update u
        if len(u.shape) == 1:
            u = torch.sum(p.clone() * torch.reshape(gauss(q,q,sigma),(npoints,1)),dim=0)
        else:
            u = torch.transpose(torch.sum(p.clone().unsqueeze(2).repeat((1,1,npoints)) * gaussVectorized(q,q,sigma).unsqueeze(1), dim=0),0,1)
            #for i in range(u.shape[0]):
            #    u[i,:] = torch.sum(p.clone() * torch.reshape(gauss(q[i,:],q,sigma),(npoints,1)),dim=0)

        # compute time derivative of p
        if len(u.shape) == 1:
            p_dot = torch.reshape(torch.mm(-torch.mm(torch.reshape(p.clone(),(3,1)),torch.reshape(d_gauss(q,q,sigma),(1,3))), torch.reshape(p.clone(),(3,1))),(1,3))
        else:
            p_dot = torch.squeeze(torch.matmul(-torch.matmul(torch.transpose(p.clone(),0,1),d_gaussVectorized(q,q,sigma).permute((2,0,1))), p.clone().unsqueeze(2)))
            #for i in range(u.shape[0]):
            #    #p_dot[i,:] = -torch.sum(p * d_gauss(q[i,:],q,sigma),dim=0) * p[i,:]
            #    p_dot[i,:] = torch.transpose(torch.mm(-torch.mm(torch.transpose(p.clone(),0,1),d_gauss(q[i,:],q,sigma)), p[i,:].clone().unsqueeze(1)),0,1)

        # update p and q
        p = p + p_dot.clone()*((nT[t]-nT[t-1]))
        q = q + u.clone()*((nT[t]-nT[t-1]))
        p_save.append(p.clone())
        q_save.append(q.clone())


    return q,p,q_save,p_save

# evolve points over simulation time
# nT is in units of years
def shootUnevenVectorizedWithRigidNoGaussNorm(q,p0,R,R_center,translation,nT,sigma,device='cpu'):
    #q=q0.clone()
    if len(q.shape) == 1:
        npoints = 1
    else:
        npoints = q.shape[0]

    # make an empty array for u, the vector field
    u = torch.zeros(q.shape).type(torch.FloatTensor).to(device=device)
    p_dot = torch.zeros(q.shape).type(torch.FloatTensor).to(device=device)

    # set gaussian sigma to some number
    #sigma=2.0

    # set some number of simulation time steps
    #nT = 10

    p=p0.clone()

    p_save = []
    q_save = []
    p_save.append(p.clone())
    q_save.append(q.clone())
    
    #gaussVq = gaussVectorized(q,q,sigma)
    #dgaussVq = d_gaussVectorized(q,q,sigma)

    # loop
    for t in range(1,len(nT)):
        # apply rigid motion
        if len(u.shape) == 1:
            q = q - R_center
            q = torch.reshape(torch.mm(R[t-1],torch.reshape(q,(3,1))) + torch.reshape(translation[:,t-1],(3,1)),(1,3))
            q = q + R_center
            p = torch.reshape(torch.mm(R[t-1],torch.reshape(p,(3,1))),(1,3))
        else:
            q = q - R_center
            q = torch.transpose(torch.mm(R[t-1],torch.transpose(q,0,1)) + torch.reshape(translation[:,t-1],(3,1)),0,1)
            q = q + R_center
            p = torch.transpose(torch.mm(R[t-1],torch.transpose(p,0,1)),0,1)
                              
        # update u
        if len(u.shape) == 1:
            u = torch.sum(p.clone() * torch.reshape(gauss(q,q,sigma),(npoints,1)),dim=0)
        else:
            u = torch.transpose(torch.sum(p.clone().unsqueeze(2).repeat((1,1,npoints)) * gaussVectorizedNoNorm(q,q,sigma).unsqueeze(1), dim=0),0,1)
            #for i in range(u.shape[0]):
            #    u[i,:] = torch.sum(p.clone() * torch.reshape(gauss(q[i,:],q,sigma),(npoints,1)),dim=0)

        # compute time derivative of p
        if len(u.shape) == 1:
            p_dot = torch.reshape(torch.mm(-torch.mm(torch.reshape(p.clone(),(3,1)),torch.reshape(d_gauss(q,q,sigma),(1,3))), torch.reshape(p.clone(),(3,1))),(1,3))
        else:
            p_dot = torch.squeeze(torch.matmul(-torch.matmul(torch.transpose(p.clone(),0,1),d_gaussVectorizedNoNorm(q,q,sigma).permute((2,0,1))), p.clone().unsqueeze(2)))
            #for i in range(u.shape[0]):
            #    #p_dot[i,:] = -torch.sum(p * d_gauss(q[i,:],q,sigma),dim=0) * p[i,:]
            #    p_dot[i,:] = torch.transpose(torch.mm(-torch.mm(torch.transpose(p.clone(),0,1),d_gauss(q[i,:],q,sigma)), p[i,:].clone().unsqueeze(1)),0,1)

        # update p and q
        p = p + p_dot.clone()*((nT[t]-nT[t-1]))
        q = q + u.clone()*((nT[t]-nT[t-1]))
        p_save.append(p.clone())
        q_save.append(q.clone())


    return q,p,q_save,p_save

# evolve points over simulation time given momentum over time pt
# nT is in units of years
def shootUnevenVectorizedWithRigidGivenPt(q,pt,R,R_center,translation,nT,sigma,device='cpu'):
    #q=q0.clone()
    if len(q.shape) == 1:
        npoints = 1
    else:
        npoints = q.shape[0]

    # make an empty array for u, the vector field
    u = torch.zeros(q.shape).type(torch.FloatTensor).to(device=device)
    
    q_save = []
    q_save.append(q.clone())
    
    #gaussVq = gaussVectorized(q,q,sigma)
    #dgaussVq = d_gaussVectorized(q,q,sigma)

    # loop
    for t in range(1,len(nT)):
        # apply rigid motion
        if len(u.shape) == 1:
            q = q - R_center
            q = torch.reshape(torch.mm(R[t-1],torch.reshape(q,(3,1))) + torch.reshape(translation[:,t-1],(3,1)),(1,3))
            q = q + R_center
            p = torch.reshape(torch.mm(R[t-1],torch.reshape(pt[t],(3,1))),(1,3))
        else:
            q = q - R_center
            q = torch.transpose(torch.mm(R[t-1],torch.transpose(q,0,1)) + torch.reshape(translation[:,t-1],(3,1)),0,1)
            q = q + R_center
            p = torch.transpose(torch.mm(R[t-1],torch.transpose(pt[t],0,1)),0,1)
                              
        # update u
        if len(u.shape) == 1:
            u = torch.sum(p.clone() * torch.reshape(gauss(q,q,sigma),(npoints,1)),dim=0)
        else:
            u = torch.transpose(torch.sum(p.clone().unsqueeze(2).repeat((1,1,npoints)) * gaussVectorized(q,q,sigma).unsqueeze(1), dim=0),0,1)
            #for i in range(u.shape[0]):
            #    u[i,:] = torch.sum(p.clone() * torch.reshape(gauss(q[i,:],q,sigma),(npoints,1)),dim=0)
        
        # update p and q
        q = q + u.clone()*((nT[t]-nT[t-1]))
        q_save.append(q.clone())

    return q, q_save

# evolve points over simulation time given momentum over time pt
# nT is in units of years
def shootUnevenVectorizedWithRigidGivenPtNoGaussNorm(q,pt,R,R_center,translation,nT,sigma,device='cpu'):
    #q=q0.clone()
    if len(q.shape) == 1:
        npoints = 1
    else:
        npoints = q.shape[0]

    # make an empty array for u, the vector field
    u = torch.zeros(q.shape).type(torch.FloatTensor).to(device=device)
    
    q_save = []
    q_save.append(q.clone())
    
    #gaussVq = gaussVectorized(q,q,sigma)
    #dgaussVq = d_gaussVectorized(q,q,sigma)

    # loop
    for t in range(1,len(nT)):
        # apply rigid motion
        if len(u.shape) == 1:
            q = q - R_center
            q = torch.reshape(torch.mm(R[t-1],torch.reshape(q,(3,1))) + torch.reshape(translation[:,t-1],(3,1)),(1,3))
            q = q + R_center
            p = torch.reshape(torch.mm(R[t-1],torch.reshape(pt[t],(3,1))),(1,3))
        else:
            q = q - R_center
            q = torch.transpose(torch.mm(R[t-1],torch.transpose(q,0,1)) + torch.reshape(translation[:,t-1],(3,1)),0,1)
            q = q + R_center
            p = torch.transpose(torch.mm(R[t-1],torch.transpose(pt[t],0,1)),0,1)
                              
        # update u
        if len(u.shape) == 1:
            u = torch.sum(p.clone() * torch.reshape(gauss(q,q,sigma),(npoints,1)),dim=0)
        else:
            u = torch.transpose(torch.sum(p.clone().unsqueeze(2).repeat((1,1,npoints)) * gaussVectorizedNoNorm(q,q,sigma).unsqueeze(1), dim=0),0,1)
            #for i in range(u.shape[0]):
            #    u[i,:] = torch.sum(p.clone() * torch.reshape(gauss(q[i,:],q,sigma),(npoints,1)),dim=0)
        
        # update p and q
        q = q + u.clone()*((nT[t]-nT[t-1]))
        q_save.append(q.clone())

    return q, q_save

# evolve points over simulation time
def shootUnevenVectorizedWithDrift(q,p0,w,dw,nT,border,sigma,device='cpu'):
    #w should be nT*x*y*z*3
    #dw should be nT*x*y*z*3*3
    #border is [min_x,max,x,min_y,max_y,min_z,max_z]
    if len(q.shape) == 1:
        npoints = 1
    else:
        npoints = q.shape[0]

    # make an empty array for u, the vector field
    u = torch.zeros(q.shape).type(torch.DoubleTensor).to(device=device)
    p_dot = torch.zeros(q.shape).type(torch.FloatTensor).to(device=device)

    # set gaussian sigma to some number
    #sigma=2.0

    # set some number of simulation time steps
    #nT = 10

    p=p0.clone()

    p_save = []
    q_save = []
    p_save.append(p.clone())
    q_save.append(q.clone())
    
    vfield = w[0].permute(3,2,1,0).unsqueeze(0).to(device=device)
    dvfield_x = dw[0,:,:,:,0,:].permute(3,2,1,0).unsqueeze(0).to(device=device)
    dvfield_y = dw[0,:,:,:,1,:].permute(3,2,1,0).unsqueeze(0).to(device=device)
    dvfield_z = dw[0,:,:,:,2,:].permute(3,2,1,0).unsqueeze(0).to(device=device)
    sample_points = q.detach().clone().unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device=device)
    scaled_points = sample_points.clone().to(device=device)                                                                              
                                                               
    #gaussVq = gaussVectorized(q,q,sigma)
    #dgaussVq = d_gaussVectorized(q,q,sigma)

    # loop
    for t in range(1,len(nT)):
        #velocity at q points at this time step
        #Here, w should be a list with length=len(nT)-1. Each element should be in the shape of x*y*z*3
        vfield = w[t-1].permute(3,2,1,0).unsqueeze(0)
        dvfield_x = dw[t-1,:,:,:,0,:].permute(3,2,1,0).unsqueeze(0)
        dvfield_y = dw[t-1,:,:,:,1,:].permute(3,2,1,0).unsqueeze(0)
        dvfield_z = dw[t-1,:,:,:,2,:].permute(3,2,1,0).unsqueeze(0)
        
        sample_points = q.detach().clone().unsqueeze(0).unsqueeze(0).unsqueeze(0)
        scaled_points = sample_points.clone()
        scaled_points[0,0,0,:,0] = (((scaled_points[0,0,0,:,0]-border[0])/(border[1]-border[0]))-0.5)*2
        scaled_points[0,0,0,:,1] = (((scaled_points[0,0,0,:,1]-border[2])/(border[3]-border[2]))-0.5)*2
        scaled_points[0,0,0,:,2] = (((scaled_points[0,0,0,:,2]-border[4])/(border[5]-border[4]))-0.5)*2
        
        velocity = torch.nn.functional.grid_sample(vfield,scaled_points,mode='bilinear',padding_mode='border')
        #dw at q points at this time stop
        dwind_x = torch.nn.functional.grid_sample(dvfield_x,scaled_points,mode='bilinear',padding_mode='border')
        dwind_y = torch.nn.functional.grid_sample(dvfield_y,scaled_points,mode='bilinear',padding_mode='border')
        dwind_z = torch.nn.functional.grid_sample(dvfield_z,scaled_points,mode='bilinear',padding_mode='border')
        
        dwind = torch.stack([dwind_x.permute(0,2,3,4,1).squeeze(),dwind_y.permute(0,2,3,4,1).squeeze(),dwind_z.permute(0,2,3,4,1).squeeze()],dim=1)
        #print(dwind.shape)
        # update u
        if len(u.shape) == 1:
            u = torch.sum(p.clone() * torch.reshape(gauss(q,q,sigma),(npoints,1)),dim=0)
        else:
            u = torch.transpose(torch.sum(p.clone().unsqueeze(2).repeat((1,1,npoints)) * gaussVectorized(q,q,sigma).unsqueeze(1), dim=0),0,1)

        # compute time derivative of p
        if len(u.shape) == 1:
            p_dot = torch.reshape(torch.mm(-torch.mm(torch.reshape(p.clone(),(3,1)),torch.reshape(d_gauss(q,q,sigma),(1,3))), torch.reshape(p.clone(),(3,1))),(1,3))
        else:
            p_dot = torch.squeeze(torch.matmul(-(torch.matmul(torch.transpose(p.clone(),0,1),(d_gaussVectorized(q,q,sigma)).permute((2,0,1)))+dwind), p.clone().unsqueeze(2)))


        # update p and q
        p = p + p_dot.clone()*((nT[t]-nT[t-1]))
        q = q + (u.clone()+velocity.permute(0,2,3,4,1).squeeze())*((nT[t]-nT[t-1]))
        p_save.append(p.clone())
        q_save.append(q.clone())


    return q,p,q_save,p_save

# evolve points over simulation time
# nT is in units of years
def shootUnevenVectorizedWithDriftNoGaussNorm(q,p0,pw0,nT,sigma,device='cpu'):
    #q=q0.clone()
    if len(q.shape) == 1:
        npoints = 1
    else:
        npoints = q.shape[0]

    # make an empty array for u, the vector field
    u = torch.zeros(q.shape).type(torch.FloatTensor).to(device=device)
    p_dot = torch.zeros(q.shape).type(torch.FloatTensor).to(device=device)

    # set gaussian sigma to some number
    #sigma=2.0

    # set some number of simulation time steps
    #nT = 10

    p=p0.clone()
    pw = pw0.clone()
    qw = q.clone()
    
    p_save = []
    q_save = []
    p_save.append(p.clone())
    q_save.append(q.clone())
    
    #gaussVq = gaussVectorized(q,q,sigma)
    #dgaussVq = d_gaussVectorized(q,q,sigma)

    # loop
    for t in range(1,len(nT)):
        # update u
        if len(u.shape) == 1:
            u = torch.sum(p.clone() * torch.reshape(gauss(q,q,sigma),(npoints,1)),dim=0)
        else:
            u = torch.transpose(torch.sum(p.clone().unsqueeze(2).repeat((1,1,npoints)) * gaussVectorizedNoNorm(q,q,sigma).unsqueeze(1), dim=0),0,1)
            # should this be evaluated at q or qw?
            w = torch.transpose(torch.sum(pw.clone().unsqueeze(2).repeat((1,1,npoints)) * gaussVectorizedNoNorm(qw,qw,sigma).unsqueeze(1), dim=0),0,1)
            #for i in range(u.shape[0]):
            #    u[i,:] = torch.sum(p.clone() * torch.reshape(gauss(q[i,:],q,sigma),(npoints,1)),dim=0)

        # compute time derivative of p
        if len(u.shape) == 1:
            p_dot = torch.reshape(torch.mm(-torch.mm(torch.reshape(p.clone(),(3,1)),torch.reshape(d_gauss(q,q,sigma),(1,3))), torch.reshape(p.clone(),(3,1))),(1,3))
        else:
            pw_dot = torch.squeeze(torch.matmul(-torch.matmul(torch.transpose(pw.clone(),0,1),d_gaussVectorizedNoNorm(qw,qw,sigma).permute((2,0,1))), p.clone().unsqueeze(2)))
            p_dot = torch.squeeze(torch.matmul(-(torch.matmul(torch.transpose(p.clone(),0,1),(d_gaussVectorizedNoNorm(q,q,sigma)).permute((2,0,1))) + torch.matmul(torch.transpose(pw.clone(),0,1),(d_gaussVectorizedNoNorm(qw,qw,sigma)).permute((2,0,1)))  ), p.clone().unsqueeze(2)))
            #for i in range(u.shape[0]):
            #    #p_dot[i,:] = -torch.sum(p * d_gauss(q[i,:],q,sigma),dim=0) * p[i,:]
            #    p_dot[i,:] = torch.transpose(torch.mm(-torch.mm(torch.transpose(p.clone(),0,1),d_gauss(q[i,:],q,sigma)), p[i,:].clone().unsqueeze(1)),0,1)

        # update p and q
        p = p + p_dot.clone()*((nT[t]-nT[t-1]))
        pw = pw + pw_dot.clone()*((nT[t]-nT[t-1]))
        q = q + (u.clone() + w)*((nT[t]-nT[t-1]))
        qw = qw + (w.clone())*((nT[t]-nT[t-1]))
        p_save.append(p.clone())
        q_save.append(q.clone())


    return q,p,q_save,p_save
