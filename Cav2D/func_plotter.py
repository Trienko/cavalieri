import numpy as np
import scipy as sp
import pylab as plt
import math

from mpl_toolkits import mplot3d

from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.pyplot as plt

def point_plot():
    # create x,y
    xx, yy = np.meshgrid(range(3), range(3))

    # calculate corresponding z
    z = 0.5 * xx

    plt3d = plt.figure().gca(projection='3d')
    # plot the surface
    plt3d.plot_surface(xx, yy, z, alpha=0.2)

    # calculate corresponding z
    z = yy

    # plot the surface
    plt3d.plot_surface(xx, yy, z, alpha=0.2)
    #plt.show()

    # calculate corresponding z
    z = -1*yy -1*xx + 4
    # plot the surface
    plt3d.plot_surface(xx, yy, z, alpha=0.2)
 

    ax = plt.gca()
    ax.hold(True)

    coordinates = np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]])   
    print(coordinates.shape)
    #ax.scatter3D(coordinates[:,0], coordinates[:,1], coordinates[:,2],c="r"); 
    #plt.show()

    #shift x by 3

    for k in range(coordinates.shape[0]):
        if coordinates[k,2] == 1:
           coordinates[k,0] += 1

   
    #shift y by 3

    for k in range(coordinates.shape[0]):
        if coordinates[k,2] == 1:
           coordinates[k,0] += 1
           coordinates[k,1] += 1 
    ax.scatter3D(coordinates[:,0], coordinates[:,1], coordinates[:,2],c="b"); 
    plt.show()
   
    #Data for a three-dimensional line
    #zline = np.linspace(0, 15, 1000)
    #xline = np.sin(zline)
    #yline = np.cos(zline)
    #ax.plot3D(xline, yline, zline, 'gray')

    # Data for three-dimensional scattered points
    #zdata = 15 * np.random.random(100)
    #xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
    #ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
    #ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens'); 
    #plt.show()


def create_xy_1(a=0,b=4,c=0,d=4,Nx = 10,Ny = 10):
    x = np.linspace(a,b,Nx)
    y = np.linspace(c,d,Ny)
    xx,yy = np.meshgrid(x,y)
    return xx,yy

def x_2_func_line(x,y,a,c,A,C,K):
    P = 1 + ((A+C)/(A*C))
    print("P = "+str(P))
    T1 = (P*A-1)/(P*A)
    print("T1 = "+str(T1))
    T2 = (P*C-1)/(P*C)
    print("T2 = "+str(T2)) 
    PA_inv = 1.0/(P*A)
    PC_inv = 1.0/(P*C)
    print("PA_inv = "+str(PA_inv))
    print("PC_inv = "+str(PC_inv))
    x2 = (x-a)*T1 - (y-c)*PA_inv + K*PA_inv
    y2 = (y-c)*T2 - (x-a)*PC_inv + K*PC_inv
    return x2,y2

def create_xy_2_line(xx, yy,a=0.0, c=0.0, A=2, C=1.0, K = 8):
    xx2 = np.zeros(xx.shape,dtype=float)
    yy2 = np.zeros(yy.shape,dtype=float)
    #print(xx2)
    #print(yy2)
    for i in range(xx.shape[0]):
        for j in range(yy.shape[1]):
            xx2[i,j],yy2[i,j] = x_2_func_line(xx[i,j],yy[i,j],a,c,A,C,K)
    return xx2,yy2

def func_p(xx2,yy2,K=8):
    zz2 = -xx2-yy2 + K
    return zz2

def plot_figures(xx,yy,xx2,yy2,zz2,K=8):
    

    #PLOT PLANE
    ############################################################
    # create x,y
    xx_new, yy_new = np.meshgrid(range(7), range(7))

    # calculate corresponding z
    z = -1*yy_new -1*xx_new + K

    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(xx2, yy2, zz2, alpha=0.2)

    #plt3d.plot_surface(xx_new, yy_new, z, alpha=0.2)

    ax = plt.gca()
    ax.hold(True)
    ############################################################
    
    #ax.scatter3D(xx2.flatten(), yy2.flatten(), zz2.flatten() ,c="b",alpha=0.2); 
    #ax.scatter3D(xx.flatten(), yy.flatten(), np.zeros((len(yy.flatten()),),dtype=float) ,c="r",alpha=0.2); 
    
    #PLOT VOLUME
    ############################################################
    volume = np.zeros((8,3),dtype=float)

    volume[0,0] = xx[0,0]
    volume[0,1] = yy[0,0]
    volume[1,0] = xx[0,-1]
    volume[1,1] = yy[0,-1]
    volume[2,0] = xx[-1,0]
    volume[2,1] = yy[-1,0]
    volume[3,0] = xx[-1,-1]
    volume[3,1] = yy[-1,-1]

    volume[4,0] = xx2[0,0]
    volume[4,1] = yy2[0,0]
    volume[4,2] = zz2[0,0]
    volume[5,0] = xx2[0,-1]
    volume[5,1] = yy2[0,-1]
    volume[5,2] = zz2[0,-1]
    volume[6,0] = xx2[-1,0]
    volume[6,1] = yy2[-1,0]
    volume[6,2] = zz2[-1,0]
    volume[7,0] = xx2[-1,-1]
    volume[7,1] = yy2[-1,-1]
    volume[7,2] = zz2[-1,-1]

    ax.plot3D(np.array([volume[0,0],volume[1,0]]), np.array([volume[0,1],volume[1,1]]), np.array([volume[0,2],volume[1,2]]), 'black') 
    ax.plot3D(np.array([volume[1,0],volume[3,0]]), np.array([volume[1,1],volume[3,1]]), np.array([volume[1,2],volume[3,2]]), 'black',ls="--") 
    ax.plot3D(np.array([volume[3,0],volume[2,0]]), np.array([volume[3,1],volume[2,1]]), np.array([volume[3,2],volume[2,2]]), 'black',ls="--") 
    ax.plot3D(np.array([volume[2,0],volume[0,0]]), np.array([volume[2,1],volume[0,1]]), np.array([volume[2,2],volume[0,2]]), 'black') 
    ax.plot3D(np.array([volume[4,0],volume[5,0]]), np.array([volume[4,1],volume[5,1]]), np.array([volume[4,2],volume[5,2]]), 'black') 
    ax.plot3D(np.array([volume[5,0],volume[7,0]]), np.array([volume[5,1],volume[7,1]]), np.array([volume[5,2],volume[7,2]]), 'black') 
    ax.plot3D(np.array([volume[7,0],volume[6,0]]), np.array([volume[7,1],volume[6,1]]), np.array([volume[7,2],volume[6,2]]), 'black') 
    ax.plot3D(np.array([volume[6,0],volume[4,0]]), np.array([volume[6,1],volume[4,1]]), np.array([volume[6,2],volume[4,2]]), 'black') 
    ax.plot3D(np.array([volume[0,0],volume[4,0]]), np.array([volume[0,1],volume[4,1]]), np.array([volume[0,2],volume[4,2]]), 'black') 
    ax.plot3D(np.array([volume[2,0],volume[6,0]]), np.array([volume[2,1],volume[6,1]]), np.array([volume[2,2],volume[6,2]]), 'black') 
    ax.plot3D(np.array([volume[3,0],volume[7,0]]), np.array([volume[3,1],volume[7,1]]), np.array([volume[3,2],volume[7,2]]), 'black') 
    ax.plot3D(np.array([volume[1,0],volume[5,0]]), np.array([volume[1,1],volume[5,1]]), np.array([volume[1,2],volume[5,2]]), 'black') 
    ############################################################
                          
    
    #DRAW INTEGRATION STRIP
    ############################################################
    delta_x = yy[1,0] - yy[0,0]
    delta_y = xx[0,1] - xx[0,0]
 
    four_points_1 = np.zeros((4,3),dtype=float)
    four_points_2 = np.zeros((4,3),dtype=float)
            
    for k in range(4):
        four_points_1[k,0] = xx[0,0]
        four_points_1[k,1] = yy[0,0]
        four_points_1[k,2] = 0 

        four_points_2[k,0] = xx2[0,0]
        four_points_2[k,1] = yy2[0,0]
        four_points_2[k,2] = zz2[0,0]

    for k in range(1,4):
          if k == 1: 
             four_points_1[k,0] += delta_x
             four_points_2[k,0] += delta_x
          if k == 2:
             four_points_1[k,1] += delta_y
             four_points_2[k,1] += delta_y
          if k == 3:
             four_points_1[k,0] += delta_x
             four_points_2[k,0] += delta_x
             four_points_1[k,1] += delta_y
             four_points_2[k,1] += delta_y

            #print(four_points_1)
            #print(four_points_2)

    Z = np.concatenate((four_points_1,four_points_2))  

    # list of sides' polygons of figure
    verts = [[Z[0],Z[1],Z[3],Z[2]],
             [Z[4],Z[5],Z[7],Z[6]], 
             [Z[0],Z[4],Z[5],Z[1]], 
             [Z[1],Z[5],Z[7],Z[3]], 
             [Z[2],Z[6],Z[7],Z[3]],
             [Z[0],Z[2],Z[6],Z[4]]]

    faces = Poly3DCollection(verts, linewidths=0.3, edgecolors='k',facecolors='cyan',alpha=0.3)
    faces.set_facecolor((0,1,0,0.1))

    ax.add_collection3d(faces)

    B = np.zeros((3,),dtype=float)
    D = np.copy(B)
    F = np.copy(B)

    #BASE 1
    #########################
    B[0] = 4.0
    
    D[0] = 4.0
    D[1] = 4.0

    F[0] = 24.0/5.0
    F[1] = 8.0/5.0
    F[2] = -F[0]-F[1]+8

    vrtx = [[B,D,F]]  
 
    faces = Poly3DCollection(vrtx, linewidths=1, edgecolors='k',facecolors='cyan',alpha=0.3)
    faces.set_facecolor((1,1,0,0.1))

    ax.add_collection3d(faces)
    #########################
    
    #BASE 2
    #########################
    V1 = np.zeros((3,),dtype=float)
    V2 = np.copy(V1)
    V3 = np.copy(V2)
    V4 = np.copy(V3)

    V1[0] = 2

    V2[0] = 2
    V2[1] = 4.0

    #2x-4 = -0.5x+4
    V3[0] = 8/2.5
    V3[1] = 2*V3[0]-4
    V3[2] = -V3[0]-V3[1]+8

    #2x = -0.5x+6
    V4[0] = (6)/2.5
    V4[1] = -0.5*V4[0]+6
    V4[2] = -V4[0]-V4[1]+8
    
    vrtx = [[V1,V2,V4,V3]]  
 
    faces = Poly3DCollection(vrtx, linewidths=1, edgecolors='k',facecolors='red',alpha=0.3)
    faces.set_facecolor((1,0,0,0.1))

    ax.add_collection3d(faces)
    #########################
    #BASE 3
    #########################

    A = np.zeros((3,),dtype=float)
    E = np.copy(A)
    G = np.copy(A)
    C = np.copy(A)

    E[0] = 8.0/5.0
    E[1] = 16.0/5.0
    E[2] = -E[0]-E[1]+8

    G[0] = 4.0/5.0
    G[1] = 28.0/5.0
    G[2] = -G[0]-G[1]+8

    C[1] = 4.0
    vrtx = [[A,E,G,C]]  
 
    faces = Poly3DCollection(vrtx, linewidths=1, edgecolors='k',facecolors='red',alpha=0.3)
    faces.set_facecolor((0,1,0,0.1))

    ax.add_collection3d(faces)
    
    #########################



    #import mpl_toolkits.mplot3d as a3
    #import matplotlib.colors as colors
    #import pylab as pl
    #import scipy as sp

    #ax = a3.Axes3D(pl.figure())
    #for i in range(10000):
    #    vtx = sp.rand(3,3)
    #    tri = a3.art3d.Poly3DCollection([vtx])
    #    tri.set_color(colors.rgb2hex(sp.rand(3)))
    #    tri.set_edgecolor('k')
    #    ax.add_collection3d(tri)
    #pl.show()

    #Draw redline --- hardcoded :-(
    ax.plot3D(np.array([xx[4,4],xx2[4,4]]), np.array([yy[4,4],yy2[4,4]]), np.array([0,zz2[4,4]]), 'red') 

    T = np.zeros((3,),dtype=float)
    T[0] = 0.8
    T[1] = 0
    T[2] = 1.6

    B = np.zeros((3,),dtype=float)
    B[0] = 4.0
    
    ax.plot3D(np.array([T[0],B[0]]), np.array([T[1]+2,B[1]+2]), np.array([T[2],B[2]]), 'blue') 


    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    plt.show()
    
    ############################################################

    
    #PLOT XY PROJ - TOP VIEW
    ############################################################
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(np.array([volume[0,0],volume[1,0]]), np.array([volume[0,1],volume[1,1]]), 'black',ls="--") 
    ax.plot(np.array([volume[1,0],volume[3,0]]), np.array([volume[1,1],volume[3,1]]), 'black',ls="--") 
    ax.plot(np.array([volume[3,0],volume[2,0]]), np.array([volume[3,1],volume[2,1]]), 'black',ls="--") 
    ax.plot(np.array([volume[2,0],volume[0,0]]), np.array([volume[2,1],volume[0,1]]), 'black',ls="--") 
    ax.plot(np.array([volume[4,0],volume[5,0]]), np.array([volume[4,1],volume[5,1]]), 'black') 
    ax.plot(np.array([volume[5,0],volume[7,0]]), np.array([volume[5,1],volume[7,1]]), 'black') 
    ax.plot(np.array([volume[7,0],volume[6,0]]), np.array([volume[7,1],volume[6,1]]), 'black') 
    ax.plot(np.array([volume[6,0],volume[4,0]]), np.array([volume[6,1],volume[4,1]]), 'black') 
    ax.plot(np.array([volume[0,0],volume[4,0]]), np.array([volume[0,1],volume[4,1]]), 'black') 
    ax.plot(np.array([volume[2,0],volume[6,0]]), np.array([volume[2,1],volume[6,1]]), 'black') 
    ax.plot(np.array([volume[3,0],volume[7,0]]), np.array([volume[3,1],volume[7,1]]), 'black') 
    ax.plot(np.array([volume[1,0],volume[5,0]]), np.array([volume[1,1],volume[5,1]]), 'black') 
    ax.plot(np.array([4.0/3.0,4.0/3.0]), np.array([16.0/3.0,8.0/3.0]), 'black',ls="-.") 
    ax.plot(np.array([1.6,1.6]), np.array([4,0]), 'black',ls="-.") 
    ax.plot(np.array([0.8,0.8]), np.array([5.6,4]), 'black',ls="-.") 
    

    #NB HARDCODED :-(
    #xy coordinates of each point
    ax.annotate('A', xy=(0, 0), xytext=(-0.12, -0.12))
    ax.annotate('B', xy=(4, 0), xytext=(4+0.06, -0.12))
    ax.annotate('C', xy=(0, 4), xytext=(-0.12, 4))
    ax.annotate('D', xy=(4, 4), xytext=(4+0.03, 4))
    ax.annotate('E', xy=(1.6, 3.2), xytext=(1.6+0.01, 3.2-0.31))
    ax.annotate('F', xy=(4.8, 1.6), xytext=(4.8+0.03, 1.6))
    ax.annotate('G', xy=(0.8, 5.6), xytext=(0.8, 5.6+0.03))
    ax.annotate('H', xy=(4.0/3.0, 16.0/3.0), xytext=(4.0/3.0, 16.0/3.0+0.03))
    ax.annotate('I', xy=(4.0/3.0, 4), xytext=(4.0/3.0+0.03, 4+0.04))
    ax.annotate('J', xy=(4.0/3.0, 8.0/3.0), xytext=(4.0/3.0+0.06, 8.0/3.0-0.04))
    ax.annotate('K', xy=(1.6, 4), xytext=(1.6, 4+0.04))
    ax.annotate('L', xy=(4, 2), xytext=(4, 2+0.04))
    ax.annotate('M', xy=(1.6, 0), xytext=(1.6+0.01, 0+0.05))
    ax.annotate('N', xy=(0.8, 4), xytext=(0.8, 4-0.3))
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    plt.show()
   

    #---VOLUME OF EACH SUBBODY---
    v = np.array([2.73067,8.256,0.5333333,3.95062,0.505679,0.293926,5.952,0.5333333,0.341333,0.606815,0.316049,1.58025])
    print(np.sum(v))

    ############################################################
    
def plot_parametric_test():
    plt3d = plt.figure().gca(projection='3d')
    t = np.linspace(1,10,100)
    x = -2*t
    y = t
    z = 4*t**2-4

    ax = plt.gca()
    ax.hold(True) 
    ax.plot3D(x, y, z, 'black')
    ax.plot3D((x+2)+10,(y-1)+10,z,'red')
    ax.scatter3D(-2,1,0)
    ax.scatter3D(10,10,0)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show() 
     
def plot_parametric_test2():
    plt3d = plt.figure().gca(projection='3d')
    t = np.linspace(0,10,200)
    x = t
    y = t
    z = np.sqrt(t)

    ax = plt.gca()
    ax.hold(True)
    ax.plot3D(x, y, z, 'black')
    #ax.plot3D((x+2)+10,(y-1)+10,z,'red')
    #ax.scatter3D(-2,1,0)
    #ax.scatter3D(10,10,0)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show() 

def create_xy_2_sphere(xx,yy,p=True):

    xx2 = np.zeros(xx.shape,dtype=float)
    yy2 = np.zeros(xx.shape,dtype=float)
    t_m = np.zeros(xx.shape,dtype=float)

    for k in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            K = -4*xx[k,j]**2 + 8*xx[k,j]*yy[k,j] + 4*xx[k,j] - 4*yy[k,j]**2+4*yy[k,j] + 33
            if p:
               t = 0.25*(-2*xx[k,j]-2*yy[k,j] - 1 + np.sqrt(K)) 
            else:
               t = 0.25*(-2*xx[k,j]-2*yy[k,j] - 1 - np.sqrt(K)) 
            t_m[k,j] = t
            xx2[k,j] = xx[k,j] + t
            yy2[k,j] = yy[k,j] + t
    
    return xx2,yy2,t_m   

def plot_volume2(xx,yy,xx2,yy2,t):
    

    #PLOT PLANE
    ############################################################
    # create x,y
    #xx_new, yy_new = np.meshgrid(np.linspace(0,4,200), np.linspace(0,4,200))

    # calculate corresponding z
    z = np.sqrt(-xx2**2-yy2**2+4)

    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(xx2, yy2, z, alpha=0.2)

    ax = plt.gca()
    ax.hold(True)

    n = np.array([0,-1])

    x_v = np.zeros((4,),dtype=float)
    y_v = np.zeros((4,),dtype=float)

    t_max = t[t.shape[0]/2,t.shape[0]/2]
    t_v = np.linspace(0,t_max,200)
    x = xx[t.shape[0]/2,t.shape[0]/2] + t_v
    y = yy[t.shape[0]/2,t.shape[0]/2] + t_v 
    z = 0 + np.sqrt(t_v)
    ax.plot3D(x,y,z,"red")        

    c = 0

    for k in n:
        for j in n:

            t_max = t[k,j]
            t_v = np.linspace(0,t_max,200)

            x = xx[k,j] + t_v
            y = yy[k,j] + t_v 
            z = 0 + np.sqrt(t_v)
            print(x[-1])
            print(y[-1])
            x_v[c] = x[-1]
            y_v[c] = y[-1]
            c = c+1   
            ax.plot3D(x,y,z,"black")
    
    #1.1861406616345072
    #1.1861406616345072 A - 0
    #1.6861406616345072
    #0.6861406616345072 B - 1
    #0.6861406616345072
    #1.6861406616345072 C - 2
    #1.3507810593582121
    #1.3507810593582121 D - 3

    for k in range(4):
        if k == 0:
           x = np.linspace(x_v[2],x_v[0],100)
           y = np.sqrt(-x**2 -x+4)
        elif k == 1:
           x = np.linspace(x_v[0],x_v[1],100)
           y = 0.5*(np.sqrt(17-4*x**2)-1)
        elif k == 2:
           x = np.linspace(x_v[3],x_v[1],100)
           y = np.sqrt(-x**2-(x-1)+4)
        else:
           x = np.linspace(x_v[2],x_v[3],100)
           y = 0.5*(np.sqrt(21-4*x**2)-1)
        z = np.sqrt(-x**2-y**2+4)
        ax.plot3D(x,y,z,"black")

    ax.plot3D(np.array([0,0]),np.array([0,1]),np.array([0,0]),"black") 
    ax.plot3D(np.array([0,1]),np.array([0,0]),np.array([0,0]),"black") 
    ax.plot3D(np.array([1,1]),np.array([1,0]),np.array([0,0]),"black",ls="--") 
    ax.plot3D(np.array([1,0]),np.array([1,1]),np.array([0,0]),"black",ls="--") 
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z$")
    
    #ax.plot3D(np.array([1,0]),np.array([1,1]),np.array([0,0]),"black") 
    #ax.plot3D(np.array([0,1]),np.array([1,1]),np.array([0,0]),"black")
    #print(xx[k,j])
    #plt.zlim(0,2)
    plt.show()
    ############################################################
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for k in range(4):
        if k == 0:
           x = np.linspace(x_v[2],x_v[0],100)
           y = np.sqrt(-x**2 -x+4)
        elif k == 1:
           x = np.linspace(x_v[0],x_v[1],100)
           y = 0.5*(np.sqrt(17-4*x**2)-1)
        elif k == 2:
           x = np.linspace(x_v[3],x_v[1],100)
           y = np.sqrt(-x**2-(x-1)+4)
        else:
           x = np.linspace(x_v[2],x_v[3],100)
           y = 0.5*(np.sqrt(21-4*x**2)-1)
        #z = np.sqrt(-x**2-y**2+4)
        ax.plot(x,y,"black")

    ax.plot(np.array([0,0]),np.array([0,1]),"black") 
    ax.plot(np.array([0,1]),np.array([0,0]),"black") 
    ax.plot(np.array([1,1]),np.array([1,0]),"black",ls="--") 
    ax.plot(np.array([1,0]),np.array([1,1]),"black",ls="--")

    ax.plot(np.array([0,x_v[3]]),np.array([0,y_v[3]]),"black")
    ax.plot(np.array([0,x_v[2]]),np.array([1,y_v[2]]),"black")
    ax.plot(np.array([1,x_v[1]]),np.array([0,y_v[1]]),"black")

    ax.plot(np.array([x_v[2],x_v[2]]),np.array([y_v[2],1]),"black",ls = "-.")
    ax.plot(np.array([1,1]),np.array([1,np.sqrt(-1**2 -1+4)]),"black",ls = "-.")
    ax.plot(np.array([x_v[0],x_v[0]]),np.array([x_v[0]-1,0.5*(np.sqrt(21-4*x_v[0]**2)-1)]),"black",ls = "-.")
    ax.plot(np.array([x_v[3],x_v[3]]),np.array([y_v[3],0.5*(np.sqrt(17-4*x_v[3]**2)-1)]),"black",ls = "-.")

    ax.annotate('A', xy=(0, 0), xytext=(-0.05, -0.05))
    ax.annotate('B', xy=(1, 0), xytext=(1+0.03, -0.03))
    ax.annotate('C', xy=(0, 1), xytext=(-0.06, 1))
    ax.annotate('D', xy=(1, 1), xytext=(1+0.03, 1-0.03))
    ax.annotate('E', xy=(x_v[0], y_v[0]), xytext=(x_v[0]+0.03, y_v[0]-0.02))
    ax.annotate('F', xy=(x_v[1], y_v[1]), xytext=(x_v[1], y_v[1]))
    ax.annotate('G', xy=(x_v[2], y_v[2]), xytext=(x_v[2], y_v[2]))
    ax.annotate('H', xy=(x_v[3], y_v[3]), xytext=(x_v[3], y_v[3]))

    ax.annotate('I', xy=(x_v[2], 1), xytext=(x_v[2], 1-0.07))

    #ax.annotate('J', xy=(1, 0.5*(np.sqrt(21-4*1**2)-1)), xytext=(1, 0.5*(np.sqrt(21-4*1**2)-1)+0.03))
    ax.annotate('J', xy=(1, np.sqrt(-1**2 -1+4)), xytext=(1+0.01, np.sqrt(-1**2 -1+4)+0.01))
    print("J = ",str(np.sqrt(-1**2 -1+4)))
    ax.annotate('K', xy=(x_v[0], 0.5*(np.sqrt(21-4*x_v[0]**2)-1)), xytext=(x_v[0], 0.5*(np.sqrt(21-4*x_v[0]**2)-1)+0.03))
    print("K = ",str(0.5*(np.sqrt(21-4*x_v[0]**2)-1)))
    ax.annotate('L', xy=(x_v[0], x_v[0]-1), xytext=(x_v[0], x_v[0]-1-0.06))
    ax.annotate('M', xy=(x_v[3], 0.5*(np.sqrt(17-4*x_v[3]**2)-1)), xytext=(x_v[3]+0.01, 0.5*(np.sqrt(17-4*x_v[3]**2)-1)))
    print("M = ",str(0.5*(np.sqrt(17-4*x_v[3]**2)-1)))
    #ax.annotate('K', xy=(1, 1), xytext=(x_v[2], 1-0.07))

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    
    print("M = ",str(x_v[0])+" "+str(x_v[0]-1))
    
    plt.show()

    #ax.scatter3D(xx2.flatten(), yy2.flatten(), zz2.flatten() ,c="b",alpha=0.2); 
    #ax.scatter3D(xx.flatten(), yy.flatten(), np.zeros((len(yy.flatten()),),dtype=float) ,c="r",alpha=0.2); 
       
    x = np.linspace(x_v[2],x_v[1],100)
    y1 = np.zeros((100,),dtype=float)
    y2 = np.zeros((100,),dtype=float)

    for k in range(len(x)):
        if (x[k]>x_v[0]):
           y1[k] = 0.5*(np.sqrt(17-4*x[k]**2)-1)
        else:
           y1[k] = np.sqrt(-x[k]**2 -x[k]+4)

        if (x[k]>x_v[3]):
           y2[k] = np.sqrt(-x[k]**2-(x[k]-1)+4)
        else:
           y2[k] = 0.5*(np.sqrt(21-4*x[k]**2)-1)

    plt.fill_between(x, y1, y2, color='red',alpha=0.2)
    plt.plot(x,y1,"black",alpha=0.2,linewidth=3)
    plt.plot(x,y2,"black",alpha=0.2,linewidth=3)
    
    x = np.array([xx[0,0],xx[0,-1],xx[-1,-1],xx[-1,0]])
    y = np.array([yy[0,0],yy[0,-1],yy[-1,-1],yy[-1,0]])
    #plt.figure(figsize=(8, 8))
    #plt.axis('equal')
    plt.fill(x, y,"b",alpha=0.2,edgecolor='black', linewidth=3)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()

    volume_v = np.array([0.2666667,0.0844182,0.0625933,0.2666667,0.0519962,0.071572,0.0234433,0.0128692,0.00547412,0.00921532,0.00912802])

    print("v = "+str(np.sum(volume_v)))
    #v = 0.8695177800000001
   
def d3(A,B):

    return np.sqrt((A[0]-B[0])**2+(A[1]-B[1])**2+(A[2]-B[2])**2)

def area_trap(a,b,c,d):
    return np.sqrt(c**2 - 0.25*(((c**2-d**2)/(b-a)) + (b-a))**2)*(a+b)*0.5

def area_heron(a,b,c):
    S = 0.5*(a+b+c)
    return np.sqrt(S*(S-a)*(S-b)*(S-c))

#COMPUTING THE VOLUME USING PRISMATOID EQUATION
def volume_of_prismatoid():
    #base 1
    #A, E, G, C

    A = np.zeros((3,),dtype=float)
    E = np.copy(A)
    G = np.copy(A)
    C = np.copy(A)

    E[0] = 8.0/5.0
    E[1] = 16.0/5.0
    E[2] = -E[0]-E[1]+8

    G[0] = 4.0/5.0
    G[1] = 28.0/5.0
    G[2] = -G[0]-G[1]+8

    C[1] = 4.0

    a = d3(G,C)
    b = d3(A,E)
    c = d3(G,E)
    d = d3(C,A)

    A1 = area_trap(a,b,c,d) 
    print("A1 = ",A1)

    #base 2
    #QRD (middle cross-section)
    Q = np.zeros((3,),dtype=float)
    R = np.copy(Q)
    D = np.copy(Q)

    Q[0] = 2.0
    
    #2x-4 = -0.5x+4
    R[0] = 8/2.5
    R[1] = 2*R[0]-4
    R[2] = -R[0]-R[1]+8

    D[0] = 4.0
    D[1] = 4.0
    
    a = d3(Q,R)
    b = d3(R,D)
    c = d3(D,Q)

    A2 = area_heron(a,b,c)
    print("A2 = ",A2)

    #base 3
    #BDF
    B = np.zeros((3,),dtype=float)
    D = np.copy(B)
    F = np.copy(B)

    B[0] = 4.0
    
    D[0] = 4.0
    D[1] = 4.0

    F[0] = 24.0/5.0
    F[1] = 8.0/5.0
    F[2] = -F[0]-F[1]+8

    a = d3(B,F)
    b = d3(F,D)
    c = d3(B,D)

    A3 = area_heron(a,b,c)
    print("A3 = ",A3)

    # vector calculations

    a_v = F-B
    b_v = D-B

    print("a_v = ",a_v)
    print("b_v = ",b_v)

    c_v = np.zeros((3,),dtype=float)
    c_v[0] = a_v[1]*b_v[2] - a_v[2]*b_v[1] 
    c_v[1] = a_v[2]*b_v[0] - a_v[0]*b_v[2]
    c_v[2] = a_v[0]*b_v[1] - a_v[1]*b_v[0]
    
    print("c_v = ",c_v)

    #B + tc_v #equation for line normal to BFD
    #z = 2x
    #c_v = ', array([-6.4,  0. ,  3.2]))
    #3.2t = 2(4-6.4t)
    #t = 0.5
    #x = 4 - 6.4(0.5) = 0.8
    #y = 0
    #z = 1.6 
    
    T = np.zeros((3,),dtype=float)
    T[0] = 0.8
    T[1] = 0
    T[2] = 1.6

    h = d3(B,T)
    print("h = ",h)

    V = (1.0/6.0)*h*(A1+4*A2+A3)
    print("V = ",V)
    
    #TESTING NEW HEIGHT IDEA
    
    V1 = np.zeros((3,),dtype=float)
    V2 = np.copy(V1)
    V3 = np.copy(V2)
    V4 = np.copy(V3)

    V1[0] = h/2.0

    V2[0] = h/2.0
    V2[1] = 4.0

    V3[0] = (h+4)/2.5
    V3[1] = -0.5*V3[0]+4
    V3[2] = -V3[0]-V3[1]+8

    V4[0] = (h+6)/2.5
    V4[1] = -0.5*V4[0]+6
    V4[2] = -V4[0]-V4[1]+8

    a = d3(V2,V4)
    b = d3(V1,V3)
    c = d3(V4,V3)
    d = d3(V1,V2)
  
    A2a = area_trap(a,b,c,d) 
    
    Va = (1.0/6.0)*h*(A1+4*A2+A3)
    print("Va = ",Va)

    #TESTING NEW NEW HEIGHT IDEA --- WORKED... IT IS  A PRISMATOID!!

    #(4,0,0)
    #(0.8,0,1.6)

    #(2.4,0,0.8) --- Midway point 

    V1 = np.zeros((3,),dtype=float)
    V2 = np.copy(V1)
    V3 = np.copy(V2)
    V4 = np.copy(V3)

    V1[0] = 2

    V2[0] = 2
    V2[1] = 4.0

    #2x-4 = -0.5x+4
    V3[0] = 8/2.5
    V3[1] = 2*V3[0]-4
    V3[2] = -V3[0]-V3[1]+8

    #2x = -0.5x+6
    V4[0] = (6)/2.5
    V4[1] = -0.5*V4[0]+6
    V4[2] = -V4[0]-V4[1]+8

    a = d3(V2,V4)
    b = d3(V1,V3)
    c = d3(V4,V3)
    d = d3(V1,V2)
  
    A2a = area_trap(a,b,c,d) 
    
    Va = (1.0/6.0)*h*(A1+4*A2a+A3)
    print("Va = ",Va)


    

    


 
     
        
   

if __name__ == "__main__":
   volume_of_prismatoid()
   xx,yy = create_xy_1(Nx = 10,Ny = 10,b=1,d=1)
   xx2,yy2,t = create_xy_2_sphere(xx,yy,p=True)
   for i in range(xx.shape[0]):
       for j in range(yy.shape[1]):
           plt.plot(xx[i,j],yy[i,j],"bo")
           plt.plot(xx2[i,j],yy2[i,j],"ro")

   #x = np.linspace(0,2,100)
   #y = np.sqrt(-x**2 -x+4)
   #y2 = np.sqrt(-x**2-(x-1)+4)
   #y3 = 0.5*(np.sqrt(17-4*x**2)-1)
   #y4 = 0.5*(np.sqrt(21-4*x**2)-1)
   #plt.plot(x,y)
   #plt.plot(x,y2)
   #plt.plot(x,y3)
   #plt.plot(x,y4)
   plt.show()
   
   plot_volume2(xx,yy,xx2,yy2,t)

   #plot_parametric_test2()
   
   xx,yy = create_xy_1()
   print(xx)
   print(yy)
   xx2,yy2 = create_xy_2_line(xx,yy)
   print(xx2)
   print(yy2)

   zz2 = func_p(xx2,yy2)

   plot_figures(xx,yy,xx2,yy2,zz2)
  
   
   #PLOTS THE DOTS (CREATE FILLED POLYGONS)???
   for i in range(xx.shape[0]):
       for j in range(yy.shape[1]):
           plt.plot(xx[i,j],yy[i,j],"bo")
           plt.plot(xx2[i,j],yy2[i,j],"ro")

   x = np.array([xx[0,0],xx[0,-1],xx[-1,-1],xx[-1,0]])
   y = np.array([yy[0,0],yy[0,-1],yy[-1,-1],yy[-1,0]])
   x2 = np.array([xx2[0,0],xx2[0,-1],xx2[-1,-1],xx2[-1,0]])
   y2 = np.array([yy2[0,0],yy2[0,-1],yy2[-1,-1],yy2[-1,0]])
   plt.show()
   #plt.figure(figsize=(8, 8))
   #plt.axis('equal')
   plt.fill(x, y,"b",x2,y2,"r",alpha=0.2,edgecolor='black', linewidth=3)
   plt.xlim(0,5.1)
   plt.xlabel("$x$")
   plt.ylabel("$y$")
   plt.show() 

  




