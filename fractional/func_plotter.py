import numpy as np
import scipy as sp
import pylab as plt
import math
#from scipy.stats import gamma

#def func_ex1(x):
#    y = (2*x+8)**3
#    return y

#def a_ex1(y):
#    x = -0.25*y**(2.0/3.0) + 9.0/2.0*y**(1.0/3.0) - 19
#    return x

#def a_ex2(y):
#    x = -0.25*y**(2.0/3.0) + 9.0/2.0*y**(1.0/3.0) - 11
#    return x

def a1(y,alpha,t,b):
    x = y - 1.0/math.gamma(alpha+1)*(t**alpha - (t-y)**alpha)
    return x+b

def g1(x,alpha,t):
    return 1.0/math.gamma(alpha+1)*(t**alpha-(t-x)**alpha)


def g_func(tau,alpha,t):
    return (1.0/math.gamma(alpha+1))*(t**alpha-(t-tau)**alpha)

def h_func(tau,alpha,t):
    return t - (t**alpha-math.gamma(alpha+1)*tau)**(1.0/alpha)

def a_func(y,alpha,t,f_inv):
    return f_inv(y) - (1.0/math.gamma(alpha+1))*(t**alpha - (t-f_inv(y))**alpha)

def f1(x):
    return x

def f1_inv(y):
    return (y);

def f2(x):
    return np.sqrt(x)

def f2_inv(y):
    return y**2

def plot_g_functions(t_vector,alpha_vector,tau_points):
    fig, ax = plt.subplots(len(t_vector), len(alpha_vector), sharex='all', sharey='all')
    for k in range(0,len(t_vector)):
        for i in range(0,len(alpha_vector)):
            tau = np.linspace(0,t_vector[k],tau_points)
            ax[k,i].plot(tau,g_func(tau,alpha_vector[i],t_vector[k])) 
            #ax[k,i].grid('on')
            if k == len(t_vector)-1:
               ax[k,i].set_xlabel(r"$\alpha$ = "+str(round(alpha_vector[i],2)))
            if i == 0:
              ax[k,i].set_ylabel("t = "+str(round(t_vector[k],1)))
    plt.show() 

def plot_h_functions(t_vector,alpha_vector,tau_points):
    fig, ax = plt.subplots(len(t_vector), len(alpha_vector), sharex='all', sharey='all')
    for k in range(0,len(t_vector)):
        for i in range(0,len(alpha_vector)):
            tau = np.linspace(0,g_func(t_vector[k],alpha_vector[i],t_vector[k]),tau_points)
            ax[k,i].plot(tau,h_func(tau,alpha_vector[i],t_vector[k])) 
            #ax[k,i].grid('on')
            if k == len(t_vector)-1:
               ax[k,i].set_xlabel(r"$\alpha$ = "+str(round(alpha_vector[i],2)))
            if i == 0:
              ax[k,i].set_ylabel("t = "+str(round(t_vector[k],1)))
    plt.show()

def plot_a_functions(t_vector,alpha_vector,y_points,f_inv,f):
    fig, ax = plt.subplots(len(t_vector), len(alpha_vector), sharex='all', sharey='all')
    for k in range(0,len(t_vector)):
        for i in range(0,len(alpha_vector)):
            y = np.linspace(0,f(t_vector[k]),y_points)
            ax[k,i].plot(a_func(y,alpha_vector[i],t_vector[k],f_inv),y) 
            #ax[k,i].grid('on')
            if k == len(t_vector)-1:
               ax[k,i].set_xlabel(r"$\alpha$ = "+str(round(alpha_vector[i],2)))
            if i == 0:
              ax[k,i].set_ylabel("t = "+str(round(t_vector[k],1)))
    plt.show() 

def plot_area(t_vector,alpha_vector,y_points,f_inv,f):
    fig, ax = plt.subplots(len(t_vector), len(alpha_vector), sharex='all', sharey='all')
    for k in range(0,len(t_vector)):
          
        for i in range(0,len(alpha_vector)):
            b = g_func(t_vector[k],alpha_vector[i],t_vector[k])
            y = np.linspace(0,f(t_vector[k]),y_points)
            x = np.linspace(0,t_vector[k],y_points)
            ax[k,i].plot(x,f(x)) 
            ax[k,i].plot(a_func(y,alpha_vector[i],t_vector[k],f_inv)+b,y,"r")  
            #ax[k,i].fill_between(x,0,f(x),color='gray',alpha=0.5)
            if k == len(t_vector)-1:
               ax[k,i].set_xlabel(r"$\alpha$ = "+str(round(alpha_vector[i],2)))
            if i == 0:
              ax[k,i].set_ylabel("t = "+str(round(t_vector[k],1)))
    plt.show() 
           

 

if __name__ == "__main__":
   num_points = 5
   t_f = 10
   alpha_f = 2    
   tau_points = 100
   y_points = 100


   t_v = np.linspace(0,t_f,num_points+1)
   t_v = t_v[1:]

   alpha_v = np.linspace(0,alpha_f,num_points+1)
   alpha_v = alpha_v[1:]

   plot_g_functions(t_v,alpha_v,tau_points)
   plot_h_functions(t_v,alpha_v,tau_points)
   plot_a_functions(t_v,alpha_v,y_points,f1_inv,f1)
   plot_area(t_v,alpha_v,y_points,f1_inv,f1)
   plot_area(t_v,alpha_v,y_points,f2_inv,f2) 
   #plot_a_functions(t_v,alpha_v,y_points,f2_inv,f2)

   '''
   fig, ax = plt.subplots(10, 10, sharex='all', sharey='all')

   print(math.gamma(1))
      
   alpha = np.linspace(0.05,2,10)
   t = np.linspace(5,10,10)
   
   for k in range(0,len(t)):
       y = np.linspace(0,t[k],100)
       for i in range(0,len(alpha)):
           
           x = a1(y,alpha[i],t[k],0)
           ax[k,i].plot(x,y)
           ax[k,i].grid('on')
           if k == len(t)-1:
              ax[k,i].set_xlabel(r"$\alpha$ = "+str(round(alpha[i],2)))
           if i == 0:
              ax[k,i].set_ylabel("t = "+str(round(t[k],1)))
           
           #if (i <> 0 and k <> 0):
           #   print(i)
           #   print(k)
           #   ax[k,i].set_xticks([])
           #   ax[k,i].set_yticks([])
   #ax[0,1].set_xticks([])
   #ax[0,1].set_yticks([])
   plt.show()

   b = g1(8,0.25,8)
   y = np.linspace(0,8,100)

   x1 = a1(y,0.25,8,0)
   x2 = a1(y,0.25,8,b)

   f = np.linspace(0,8,100)


   plt.plot(x1,y)
   plt.plot(x2,y)
   plt.plot(f,f)
   plt.show()
 
   x = np.linspace(0,b,100)
   y = g1(x,1.5,8)

   plt.plot(x,y)
   plt.show()


   #print(b) 
   '''

