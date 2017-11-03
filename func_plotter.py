import numpy as np
import scipy as sp
import pylab as plt

def func_ex1(x):
    y = (2*x+8)**3
    return y

def a_ex1(y):
    x = -0.25*y**(2.0/3.0) + 9.0/2.0*y**(1.0/3.0) - 19
    return x

def a_ex2(y):
    x = -0.25*y**(2.0/3.0) + 9.0/2.0*y**(1.0/3.0) - 11
    return x



if __name__ == "__main__":
   import matplotlib
   matplotlib.rcParams.update({'font.size': 16})
   fig,ax = plt.subplots()
   x = np.linspace(-19,9,1000)
   y = func_ex1(x) 
   ax.plot(x,y,'k')
   y1 = np.linspace(0,3000,5000)
   x1 = a_ex1(y1)
   x2 = a_ex2(y1)
   ax.plot(x1,y1,'k')
   x1 = a_ex1(y1)
   ax.plot(x2,y1,'k')
   plt.title("$f(x) = (2x+8)^3$")

   ax.annotate('$f(x)$',
            xy=(0, 520), xycoords='data',
            xytext=(-20,0), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='right', verticalalignment='bottom') 

   ax.annotate('$a(y)$',
            xy=(-2, 2000), xycoords='data',
            xytext=(-20,0), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='right', verticalalignment='bottom') 

   ax.annotate('$b(y)$',
            xy=(6, 2000), xycoords='data',
            xytext=(20,0), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='left', verticalalignment='bottom') 

   ax.annotate('$a$',
            xy=(-19, 0), xycoords='data',
            xytext=(0,20), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='right', verticalalignment='bottom')

   ax.annotate('$b$',
            xy=(-11, 0), xycoords='data',
            xytext=(0,20), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='right', verticalalignment='bottom')

   ax.annotate("$a'$",
            xy=(1, 1000), xycoords='data',
            xytext=(-20,0), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='right', verticalalignment='bottom')

   ax.annotate("$b'$",
            xy=(3, 2730), xycoords='data',
            xytext=(20,0), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='left', verticalalignment='bottom')
 
  
   #x_new = x<
   #ax.fill_between(x, 1000, y, where=y >= 1000)

   plt.ylim([-1,3000])
   plt.xlabel("$x$")
   plt.ylabel("$y$")
   plt.show()


