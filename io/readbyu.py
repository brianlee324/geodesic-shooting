import numpy as np
import scipy as sp

# Reads from .byu file
def readbyu(byufile):
    with open(byufile,'r') as fbyu:
        ln0 = fbyu.readline()
        ln = ln0.split()
        # read header
        ncomponents = int(ln[0])	# number of components
        npoints = int(ln[1])  # number of vertices
        nfaces = int(ln[2]) # number of faces
                    #fscanf(fbyu,'%d',1);		% number of edges
                    #%ntest = fscanf(fbyu,'%d',1);		% number of edges
        for k in range(ncomponents):
            fbyu.readline() # components (ignored)
        # read data
        vertices = np.empty([npoints, 3])
        k=-1
        while k < npoints-1:
            ln = fbyu.readline().split()
            k=k+1 ;
            vertices[k, 0] = float(ln[0]) 
            vertices[k, 1] = float(ln[1]) 
            vertices[k, 2] = float(ln[2])
            if len(ln) > 3:
                k=k+1 ;
                vertices[k, 0] = float(ln[3])
                vertices[k, 1] = float(ln[4]) 
                vertices[k, 2] = float(ln[5])

        faces = np.empty([nfaces, 3])
        ln = fbyu.readline().split()
        kf = 0
        j = 0
        while ln:
            if kf >= nfaces:
                break
            for s in ln:
                faces[kf,j] = int(sp.fabs(int(s)))
                j = j+1
                if j == 3:
                    kf=kf+1
                    j=0
            ln = fbyu.readline().split()
    faces = np.int_(faces) - 1
    xDef1 = vertices[faces[:, 0], :]
    xDef2 = vertices[faces[:, 1], :]
    xDef3 = vertices[faces[:, 2], :]
    centers = (xDef1 + xDef2 + xDef3) / 3
    surfel =  np.cross(xDef2-xDef1, xDef3-xDef1)
    return faces, vertices, centers, surfel
