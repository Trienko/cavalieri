import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random

def f_x(x,y):
    return x**2

def g_x(x,z):
  return x

fig = plt.figure()

x_max = 3.0
d = 0.05
y_max = g_x(x_max,0)
z_max = f_x(x_max,0)

#PLOT SURFACES
#PLOT f_x
ax = fig.add_subplot(111, projection='3d')
x = np.arange(0, x_max, d)
y = np.arange(0, y_max, d)
X, Y = np.meshgrid(x, y)

zs = np.array([f_x(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z,alpha=0.2)

#PLOT g_x
x = np.arange(0, x_max, d)
z = np.arange(0, z_max, d)
X, Z = np.meshgrid(x, z)
ys = np.array([g_x(x,z) for x,z in zip(np.ravel(X), np.ravel(Z))])
Y = ys.reshape(X.shape)
ax.plot_surface(X, Y, Z,color="red",alpha=0.2)

#PLOT CURVES
x = np.arange(0, x_max, d)
#PLOT PROJECTION
ax.plot(np.zeros(x.shape),g_x(x,0),f_x(x,0),color="black")

#PLOT INTERSECTION
ax.plot(x,g_x(x,0),f_x(x,0),color="black",linestyle='dashed',linewidth=1)

import matplotlib 
#matplotlib.rc('xtick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=20) 
matplotlib.rcParams.update({'font.size': 60})
#matplotlib.rc('xlabel', labelsize=20)
#matplotlib.rc('ylabel', labelsize=20)
#matplotlib.rc('zlabel', labelsize=20)

ax.set_xlabel('$x$')
ax.set_ylabel('$g(x)$')
ax.set_zlabel('$f(x)$')

ax.view_init(30,220)

plt.show()

'''
def fun(x, y):
  return x**2

def fun2(x, z):
  return x

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(0, 3.0, 0.05)
X, Y = np.meshgrid(x, y)

zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
ax.plot_surface(X, Y, Z,alpha=0.2)


x = np.arange(0, 3.0, 0.05)
z = np.arange(0, 9.0, 0.05)
X, Z = np.meshgrid(x, z)
ys = np.array([fun2(x,z) for x,z in zip(np.ravel(X), np.ravel(Z))])
Y = ys.reshape(X.shape)
ax.plot_surface(X, Y, Z,color="red",alpha=0.2)

#cset = ax.contour(X, Y, Z, zdir='x', offset=0, cmap="jet")
x = np.arange(0, 3.0, 0.05)
#z = np.arange(0, 9.0, 0.05)
ax.plot(np.zeros(x.shape),x,x**2,color="black")

ax.plot(x,x,x**2,color="black",linestyle='dashed',linewidth=1)

#ax.plot(x,x**2,c="k")

ax.set_xlabel('x')
ax.set_ylabel('g(x)')
ax.set_zlabel('f(x)')

ax.view_init(30,220)

plt.show()
'''
