import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import math
from matplotlib.collections import PolyCollection
from matplotlib.colors import colorConverter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def x_t(t,v = 200):
    x = v*t
    return x

def a_x(x,c=3e8,v = 200):
    l = np.sqrt(1 - v**2/c**2)
    a = v**(-1)*x - (v**(-1)*x)/l
    return a

t = np.linspace(0,1,200)




xt = x_t(t,v=2.999999e8)

x = np.linspace(0,xt[-1],1000)

a = a_x(x,v=2.999999e8)

plt.plot(t,xt)

print(a)
plt.plot(a,x)

plt.show()
