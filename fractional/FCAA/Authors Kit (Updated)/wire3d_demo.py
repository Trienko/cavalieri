import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import math
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def f_x(x,y):
    return np.sqrt(x)

def g_x(x,z):
  return x

def g_x_2(x,z,t=5,a=0.5):
    return (t**(a)-(t-x)**(a))/math.gamma(a+1)

import matplotlib 
#plt.locator_params(nbins=4)

fig = plt.figure()
matplotlib.rcParams.update({'font.size': 22})

counter = 1

alpha = np.array([0.3,0.6,0.9])
t_v = np.array([3,6,9])
d = 0.01

for i in range(3):
   for j in range(3):
	ax = fig.add_subplot(3,3,counter, projection='3d')	
        x_max = t_v[j]
        y_max = g_x_2(x_max,0,t=t_v[j],a=alpha[i])
        z_max = f_x(x_max,0)

        #PLOT SURFACES

        #PLOT g_x
        x = np.arange(0, x_max, d)
        z = np.arange(0, z_max, d)
        X, Z = np.meshgrid(x, z)
        ys = np.array([g_x_2(x,z,t=t_v[j],a=alpha[i]) for x,z in zip(np.ravel(X), np.ravel(Z))])
        Y = ys.reshape(X.shape)
        ax.plot_surface(X, Y, Z,color="blue",alpha=0.2,linewidth=0)

        #PLOT f_x
        x = np.arange(0, x_max, d)
        y = np.arange(0, y_max, d)
        X, Y = np.meshgrid(x, y)
        zs = np.array([f_x(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Z = zs.reshape(X.shape)
        ax.plot_surface(X, Y, Z,color="cyan",alpha=0.2,linewidth=0)


        #PLOT CURVES
        x = np.arange(0, x_max, d)

        #PLOT PROJECTION
        ax.plot(np.zeros(x.shape),g_x_2(x,0,t=t_v[j],a=alpha[i]),f_x(x,0),color="black")
        #PLOT INTERSECTION
        ax.plot(x,g_x_2(x,0,t=t_v[j],a=alpha[i]),f_x(x,0),color="black",linestyle='dashed',linewidth=1.5)

        xs = np.zeros(x.shape)
        xs = np.append(xs,np.array([0]))
        ys =  g_x_2(x,0,t=t_v[j],a=alpha[i])
        ys = np.append(ys,np.array([ys[-1]]))
        zs = f_x(x,0)
        zs = np.append(zs,np.array([0]))


        def cc(arg):
        	return colorConverter.to_rgba(arg, alpha=0.2)

	
        #ADD SHADOW
        verts = [list(zip(xs, ys, zs))]
	#print(verts)
	poly = Poly3DCollection(verts,facecolor=cc('r'))
	poly.set_alpha(0.2)
	ax.add_collection3d(poly)
	#plt.show()

	xs = x
	xs = np.append(xs,xs[::-1])
	ys =  g_x_2(x,0,t=t_v[j],a=alpha[i])
	ys = np.append(ys,ys[::-1])
	zs = f_x(x,0)
	zs = np.append(zs,np.zeros(x.shape))
	
        #ADD FENCE
        verts = [list(zip(xs, ys, zs))]
	#print(verts)
	poly = Poly3DCollection(verts,facecolor=cc('g'),alpha=0.5)
	#poly.set_alpha(0.2)
	ax.add_collection3d(poly)

	
	#matplotlib.rc('xtick', labelsize=20) 
	#matplotlib.rc('ytick', labelsize=20) 
	#matplotlib.rc('xlabel', labelsize=20)
	#matplotlib.rc('ylabel', labelsize=20)
	#matplotlib.rc('zlabel', labelsize=20)
        tick_spacing = 1
        import matplotlib.ticker as ticker
        ax.view_init(30,200)


        if counter == 2:
           ax.set_xlabel(r'$\tau$',labelpad=10)
	   ax.set_ylabel(r'$g_t^{\alpha}(\tau)$',labelpad=18)
	   ax.set_zlabel(r'$f(\tau)$')

        if counter <=3:
           ax.set_title(r"$t = $"+str(t_v[counter-1]))

        if counter == 1 or counter ==4 or counter == 7:
           ax.set_zlabel(r'$\alpha =$'+str(alpha[i]))


        if j == 2: 
           ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing+2))
        else:
           ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing+1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
	ax.zaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
	
        

	ax.view_init(30,200)
        counter = counter + 1 
#plt.tight_layout(h_pad=0.5,w_pad=0.5,pad=0.5)
plt.show()


#xs=x
#xs[-1]= 10
#verts = [list(zip(xs, ys, zs))]
#ax.add_collection3d(Poly3DCollection(verts), zs='z')


#ax.fill_between(g_x_2(x,0),f_x(x,0))
#cset = ax.contourf(np.zeros(x.shape), g_x_2(x,0), f_x(x,0), zdir='z', offset=-100)

#def cc(arg):
#    return colorConverter.to_rgba(arg, alpha=0.6)

#xs = g_x_2(x,0)
#verts = []
#zs = [0.0]
#for z in zs:
#    ys = f_x(x,0)
#    ys[0], ys[-1] = 0, 0
#    verts.append(list(zip(xs, ys)))

#poly = PolyCollection(verts, facecolors=[cc('r'), cc('g'), cc('b'),cc('y')])
#poly.set_alpha(0.7)
#ax.add_collection3d(poly, zs=zs, zdir='y')



import matplotlib 
#matplotlib.rc('xtick', labelsize=20) 
#matplotlib.rc('ytick', labelsize=20) 
matplotlib.rcParams.update({'font.size': 60})
#matplotlib.rc('xlabel', labelsize=20)
#matplotlib.rc('ylabel', labelsize=20)
#matplotlib.rc('zlabel', labelsize=20)

ax.set_xlabel(r'$\tau$')
ax.set_ylabel(r'$g_t^{\alpha}(\tau)$')
ax.set_zlabel(r'$f(\tau)$')

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
