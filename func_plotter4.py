import numpy as np
import scipy as sp
import pylab as plt
import scipy.special

def func_ex1(x):
    y = x
    return y

def func_g_y(y,mu,sig):
    x = 0.5*(1+scipy.special.erf((y-mu)/sig))
    return x

def a_ex1(y,mu=0.0,sig=1.0,a_p=0.5):
    x = y - func_g_y(y,mu,sig) + func_g_y(a_p,mu,sig) 
    return x

def a_ex2(y,c,mu=0.0,sig=1.0,a_p=0.5):
    x = y - func_g_y(y,mu,sig) + func_g_y(a_p,mu,sig) + c 
    return x

#
#def a_ex2(y):
#    x = np.arcsin(y) - np.sqrt(1-y**2) + 1
#    return x

if __name__ == "__main__":
   a_0 = a_ex1(0,mu=0.0,sig=1.0,a_p=0.5)
   print "a_0 = ",a_0

   c = func_g_y(1.0,0,1.0) - func_g_y(0.5,0,1.0) 
   print "c = ",c

   C = a_0 - func_g_y(0.5,0,1.0)
   print "C = ",C

   fig,ax = plt.subplots()
   x = np.linspace(0,1.5,1000)
   y = func_ex1(x) 
   ax.plot(x,y,'k')
   y1 = np.linspace(-0.5,1.5,5000)
   #y2 = np.linspace(0.4,4,5000)
   x1 = a_ex1(y1)
   print "x1 = ",x1
   x2 = a_ex2(y1,c)
   ax.plot(x1,y1,'k')
   #ax.plot(x1,y1,'k')
   #x1 = a_ex1(y1)
   ax.plot(x2,y1,'k')
   #plt.title("$f(x) = sin(x)$")

   '''
   ax.annotate('$f(x)$',
            xy=(0, 520), xycoords='data',
            xytext=(-15,0), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='right', verticalalignment='bottom') 

   ax.annotate('$a(y)$',
            xy=(-2, 2000), xycoords='data',
            xytext=(-15,0), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='right', verticalalignment='bottom') 

   ax.annotate('$b(y)$',
            xy=(6, 2000), xycoords='data',
            xytext=(15,0), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='left', verticalalignment='bottom') 

   ax.annotate('$a$',
            xy=(-19, 0), xycoords='data',
            xytext=(0,15), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='right', verticalalignment='bottom')

   ax.annotate('$b$',
            xy=(-11, 0), xycoords='data',
            xytext=(0,15), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='right', verticalalignment='bottom')

   ax.annotate("$a'$",
            xy=(1, 1000), xycoords='data',
            xytext=(-15,0), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='right', verticalalignment='bottom')

   ax.annotate("$b'$",
            xy=(3, 2730), xycoords='data',
            xytext=(15,0), textcoords='offset points',
            arrowprops=dict(arrowstyle="->",facecolor='black'),
            horizontalalignment='left', verticalalignment='bottom')
   '''
  
   #x_new = x<
   #ax.fill_between(x, 1000, y, where=y >= 1000)

   #plt.ylim([-1,3000])
   plt.xlabel("$x$")
   plt.ylabel("$y$")
   plt.show()


