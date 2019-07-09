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
    T1 = (P*A-1)/(P*A)
    T2 = (P*C-1)/(P*C)
    PA_inv = 1.0/(P*A)
    PC_inv = 1.0/(P*C)
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

def test(xx,yy,xx2,yy2,zz2,K=8):
    # create x,y
    xx_new, yy_new = np.meshgrid(range(7), range(7))

    # calculate corresponding z
    z = -1*yy_new -1*xx_new + K

    plt3d = plt.figure().gca(projection='3d')
    plt3d.plot_surface(xx_new, yy_new, z, alpha=0.2)

    ax = plt.gca()
    ax.hold(True)
    #ax.scatter3D(xx2.flatten(), yy2.flatten(), zz2.flatten() ,c="b",alpha=0.2); 
    #ax.scatter3D(xx.flatten(), yy.flatten(), np.zeros((len(yy.flatten()),),dtype=float) ,c="r",alpha=0.2); 

    delta_x = yy[1,0] - yy[0,0]
    delta_y = xx[0,1] - xx[0,0]


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
                          

    print(delta_x)
    print(delta_y)

    #z_order = xx.shape[0]*zz.shape[1]

    t1 = range(xx.shape[0])
    t2 = range(yy.shape[1])
    t1 = t1[::-1]
    t2 = t2[::-1]

    for j in range(1,2):
        for i in range(4,5):

            four_points_1 = np.zeros((4,3),dtype=float)
            four_points_2 = np.zeros((4,3),dtype=float)
            
            for k in range(4):
                four_points_1[k,0] = xx[i,j]
                four_points_1[k,1] = yy[i,j]
                four_points_1[k,2] = 0 

                four_points_2[k,0] = xx2[i,j]
                four_points_2[k,1] = yy2[i,j]
                four_points_2[k,2] = zz2[i,j]

            for k in range(1,4):
                print(k)
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

            if (i == 3):
               if (j == 3):
                  faces = Poly3DCollection(verts, linewidths=0.3, edgecolors='k',facecolors='red',alpha=0.3)
               else:
                  faces = Poly3DCollection(verts, linewidths=0.3, edgecolors='k',facecolors='cyan',alpha=0.3)
            else:
                faces = Poly3DCollection(verts, linewidths=0.3, edgecolors='k',facecolors='cyan',alpha=0.3)
            faces.set_facecolor((0,1,0,0.1))

            ax.add_collection3d(faces)

            # plot sides
            #ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

            '''
            ax.plot3D(np.array([four_points_1[0,0],four_points_1[1,0]]), np.array([four_points_1[0,1],four_points_1[1,1]]), np.array([four_points_1[0,2],four_points_1[1,2]]), 'black',alpha=0.3) 
                         
            ax.plot3D(np.array([four_points_1[0,0],four_points_1[2,0]]), np.array([four_points_1[0,1],four_points_1[2,1]]), np.array([four_points_1[0,2],four_points_1[2,2]]), 'black',alpha=0.3) 

            ax.plot3D(np.array([four_points_1[1,0],four_points_1[3,0]]), np.array([four_points_1[1,1],four_points_1[3,1]]), np.array([four_points_1[1,2],four_points_1[3,2]]), 'black',alpha=0.3) 

            ax.plot3D(np.array([four_points_1[2,0],four_points_1[3,0]]), np.array([four_points_1[2,1],four_points_1[3,1]]), np.array([four_points_1[2,2],four_points_1[3,2]]), 'black',alpha=0.3) 

            ax.plot3D(np.array([four_points_2[0,0],four_points_2[1,0]]), np.array([four_points_2[0,1],four_points_2[1,1]]), np.array([four_points_2[0,2],four_points_2[1,2]]), 'black',alpha=0.3) 
                         
            ax.plot3D(np.array([four_points_2[0,0],four_points_2[2,0]]), np.array([four_points_2[0,1],four_points_2[2,1]]), np.array([four_points_2[0,2],four_points_2[2,2]]), 'black',alpha=0.3) 

            ax.plot3D(np.array([four_points_2[1,0],four_points_2[3,0]]), np.array([four_points_2[1,1],four_points_2[3,1]]), np.array([four_points_2[1,2],four_points_2[3,2]]), 'black',alpha=0.3) 

            ax.plot3D(np.array([four_points_2[2,0],four_points_2[3,0]]), np.array([four_points_2[2,1],four_points_2[3,1]]), np.array([four_points_2[2,2],four_points_2[3,2]]), 'black',alpha=0.3) 

            for k in range(4):
                ax.plot3D(np.array([four_points_1[k,0],four_points_2[k,0]]), np.array([four_points_1[k,1],four_points_2[k,1]]), np.array([four_points_1[k,2],four_points_2[k,2]]), 'gray',alpha=0.3) 
            '''

    #for i in range(len(xx)):
    #    for j in range(len(yy)):
    #        ax.plot3D(np.array([xx[i,j],xx2[i,j]]), np.array([yy[i,j],yy2[i,j]]), np.array([0,zz2[i,j]]), 'black',alpha=0.3)


    plt.show()
    
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


    #x1 = np.linspace(0,2,200)
    #x2 = np.linspace(0,4,200)

    #plt.plot(x1,2*x1,"b")
    #plt.plot(x2,-0.5*x2 + 4,"r")

    print(np.array([volume[4,0],volume[4,1],volume[4,2]]))
    print((np.array([volume[5,0],volume[5,1],volume[5,2]])))
    print((np.array([volume[6,0],volume[6,1],volume[6,2]])))
    print((np.array([volume[7,0],volume[7,1],volume[7,2]])))
    plt.show()
   
 

if __name__ == "__main__":
   xx,yy = create_xy_1()
   print(xx)
   print(yy)
   xx2,yy2 = create_xy_2_line(xx,yy)
   print(xx2)
   print(yy2)

   zz2 = func_p(xx2,yy2)

   test(xx,yy,xx2,yy2,zz2)
  
   for i in range(xx.shape[0]):
       for j in range(yy.shape[1]):
           plt.plot(xx[i,j],yy[i,j],"bo")
           plt.plot(xx2[i,j],yy2[i,j],"ro")

   plt.show()

   #point_plot()
   
   




