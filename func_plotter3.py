import numpy as np
import scipy as sp
import pylab as plt

def func_ex1(x):
    y = np.sin(x)
    return y

def a_ex1(y):
    x = np.arcsin(y) - np.sqrt(1-y**2)
    return x

def a_ex2(y):
    x = np.arcsin(y) - np.sqrt(1-y**2) + 1
    return x



if __name__ == "__main__":
   import matplotlib
   matplotlib.rcParams.update({'font.size': 16})
   fig,ax = plt.subplots()
   x = np.linspace(-np.pi/2,0,1000)
   y = func_ex1(x) 
   ax.plot(x,y,'k')
   y1 = np.linspace(-1,0,5000)
   #y2 = np.linspace(0.4,4,5000)
   x1 = a_ex1(y1)
   x2 = a_ex2(y1)
   ax.plot(x1,y1,'k')
   x1 = a_ex1(y1)
   ax.plot(x2,y1,'k')
   plt.title("$f(x) = sin(x)$")

   
   ax.annotate('$f(x)$',
            xy=(-0.90, -0.79), xycoords='data',
            xytext=(25,-10), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='left', verticalalignment='bottom') 

   ax.annotate('$a(y)$',
            xy=(-1.32, -0.39), xycoords='data',
            xytext=(-25,0), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='right', verticalalignment='bottom') 

   ax.annotate('$b(y)$',
            xy=(-0.52, -0.79), xycoords='data',
            xytext=(25,0), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='left', verticalalignment='bottom') 

   ax.annotate('$a$',
            xy=(-1, 0), xycoords='data',
            xytext=(-30,0), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='left', verticalalignment='bottom')

   ax.annotate("$b',b$",
            xy=(0, 0), xycoords='data',
            xytext=(0,-30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='left', verticalalignment='bottom')

   ax.annotate("$a'$",
            xy=(-1.57, -1), xycoords='data',
            xytext=(35,-16), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='right', verticalalignment='bottom')

   #ax.annotate("$b',b$",
   #         xy=(0, 0), xycoords='data',
   #         xytext=(-25,4), textcoords='offset points',
   #         arrowprops=dict(arrowstyle="->",facecolor='black'),
   #         horizontalalignment='left', verticalalignment='bottom')
   
  
   #x_new = x<
   #ax.fill_between(x, 1000, y, where=y >= 1000)

   #plt.ylim([-1,3000])
   plt.xlabel("$x$")
   plt.ylabel("$y$")
   plt.show()


