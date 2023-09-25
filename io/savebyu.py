#Saves in .byu format
def savebyu(faces, vertices, byufile):
    #FV = readbyu(byufile)
    #reads from a .byu file into matlab's face vertex structure FV

    with open(byufile,'w') as fbyu:
        # copy header
        ncomponents = 1         # number of components
        npoints = vertices.shape[0] # number of vertices
        nfaces = faces.shape[0]                # number of faces
        nedges = 3*nfaces           # number of edges

        str = '{0: d} {1: d} {2: d} {3: d} \n'.format(ncomponents, npoints, nfaces,nedges)
        fbyu.write(str)
        str = '1 {0: d}\n'.format(nfaces)
        fbyu.write(str)


        k=-1
        while k < (npoints-1):
            k=k+1
            str = '{0: f} {1: f} {2: f} '.format(vertices[k, 0], vertices[k, 1], vertices[k, 2])
            fbyu.write(str)
            if k < (npoints-1):
                k=k+1
                str = '{0: f} {1: f} {2: f}\n'.format(vertices[k, 0], vertices[k, 1], vertices[k, 2])
                fbyu.write(str)
            else:
                fbyu.write('\n')

        j = 0
        for k in range(nfaces):
            for kk in (0,1):
                fbyu.write('{0: d} '.format(faces[k,kk]+1))
                j=j+1
                if j==3:
                    fbyu.write('\n')
                    j=0

            fbyu.write('{0: d} '.format(-faces[k,2]-1))
            j=j+1
            if j==3:
                fbyu.write('\n')
                j=0
