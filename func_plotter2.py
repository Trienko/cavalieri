import numpy as np
import scipy as sp
import pylab as plt

def func_ex1(x):
    y = 1/x
    return y

def a_ex1(y):
    x = (y**2+y-1)/y**2
    return x

def a_ex2(y):
    x = (y**2+y-1)/y**2 + 3
    return x



if __name__ == "__main__":
   import matplotlib
   matplotlib.rcParams.update({'font.size': 16})
   fig,ax = plt.subplots()
   x = np.linspace(0.25,4,1000)
   y = func_ex1(x) 
   ax.plot(x,y,'k')
   y1 = np.linspace(0.4,4,5000)
   y2 = np.linspace(0.4,4,5000)
   x1 = a_ex1(y1)
   x2 = a_ex2(y2)
   ax.plot(x1,y1,'k')
   x1 = a_ex1(y1)
   ax.plot(x2,y1,'k')
   plt.xlim([-2.5,4.5])
   plt.title("$f(x) = 1/x$")

   
   ax.annotate('$f(x)$',
            xy=(0.5, 1.98), xycoords='data',
            xytext=(-20,0), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='right', verticalalignment='bottom') 

   ax.annotate('$a(y)$',
            xy=(1.21, 3.25), xycoords='data',
            xytext=(-20,0), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='right', verticalalignment='bottom') 

   ax.annotate('$b(y)$',
            xy=(4.2, 3.33), xycoords='data',
            xytext=(-20,0), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='right', verticalalignment='bottom') 

   #ax.annotate('$a$',
   #         xy=(-19, 0), xycoords='data',
   #         xytext=(0,15), textcoords='offset points',
   #         arrowprops=dict(arrowstyle="->",facecolor='black'),
   #         horizontalalignment='right', verticalalignment='bottom')

   #ax.annotate('$b$',
   #         xy=(-11, 0), xycoords='data',
   #         xytext=(0,15), textcoords='offset points',
   #         arrowprops=dict(arrowstyle="->",facecolor='black'),
   #         horizontalalignment='right', verticalalignment='bottom')

   ax.annotate("$a'$",
            xy=(1, 1), xycoords='data',
            xytext=(-20,0), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='right', verticalalignment='bottom')

   ax.annotate("$b'$",
            xy=(2, 0.5), xycoords='data',
            xytext=(15,-25), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='left', verticalalignment='bottom')
   
  
   #x_new = x<
   #ax.fill_between(x, 1000, y, where=y >= 1000)

   #plt.ylim([-1,3000])
   plt.xlabel("$x$")
   plt.ylabel("$y$")
   plt.show()


