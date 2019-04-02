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

def plt_gh_functions(alpha = np.array([0.2,0.4,0.6,0.8,1.0]),t=10,num=1000):
    tau = np.linspace(0,t,num)
    #[0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0]
    v = np.linspace(0,1.5,len(alpha))

    for k in range(len(alpha)):
        tau2 = np.linspace(0,(t**alpha[k])/math.gamma(alpha[k]+1),num)
        y1 = g_func(tau,alpha[k],t)
        y2 = h_func(tau2,alpha[k],t)
        col = [v[k]/2,v[k]/2,v[k]/2]    
        plt.plot(tau,y1,color=col)
        plt.plot(tau2,y2,color=col,ls = "--")

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$y$")
    plt.xlim([0,10])
    plt.ylim([0,10])
    plt.show()



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

def plot_b_functions(t_vector=np.array([2,4,6,8,10]),alpha_vector=np.array([0.2,0.4,0.6,0.8]),y_points=50000,f_inv=f1_inv,f=f1):
    #plt.grid('on')
    fig,ax = plt.subplots(1,len(alpha_vector),sharex='all', sharey='all')
    #plt.grid('on')
    v = np.linspace(0,1.5,len(t_vector))

    for i in range(len(alpha_vector)):
        x = np.linspace(0,t_vector[-1],y_points)
        
        ax[i].plot(x,f(x),"k") 
        if i == 0:
           ax[i].set_xlabel(r"$\tau$")
           ax[i].set_ylabel(r"$y$")
        ax[i].set_title(r"$\alpha = $"+str(alpha_vector[i]))
        for k in range(len(t_vector)):
            y = np.linspace(0,f(t_vector[k]),y_points)
            b = g_func(t_vector[k],alpha_vector[i],t_vector[k])
            b_p = h_func(b,alpha_vector[i],t_vector[-1])
            y2 = np.linspace(0,f(b_p),y_points)
            col = [v[k]/2,v[k]/2,v[k]/2]    
        
            ax[i].plot(a_func(y,alpha_vector[i],t_vector[k],f_inv)+b,y,color=col) 
            if i == len(alpha_vector)-1:
               ax[i].plot(a_func(y2,alpha_vector[i],t_vector[-1],f_inv)+b,y2,color=col,ls="--") 
            ax[i].set_xlim([0,t_vector[-1]+0.6])
            ax[i].set_ylim([0,f(t_vector[-1])]) 
              
    plt.show()        
            
def scaling(t_vector=np.array([2,4,6,8,10]),alpha=0.8,y_points=1000,f_inv=f1_inv,f=f1):
    
    for k in range(2,len(t_vector)):
            y = np.linspace(0,f(t_vector[k]),y_points)
            s = a_func(y,alpha,t_vector[k],f_inv)/a_func(y,alpha,t_vector[k-1],f_inv)
            y = np.linspace(0,f(t_vector[k]),y_points)
            plt.plot(s)    
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

def plot_gamma(x):
    y = np.zeros((len(x),))

    for k in range(len(x)):
        #print(k)
        #print(x[k])
        try:
           # put the code you want to try here
           y[k] = math.gamma(x[k])
        except ValueError:
           # what to do if we get a value error
           y[k] = np.NaN
    
    plt.plot(x,y) 
    plt.ylim([-5,5])      
    plt.show()




def plot_gamma2(x):
    
    x = np.linspace(-5,-4,1000)
    x = x[1:-2]
    y = np.zeros((len(x),))

    for k in range(len(x)):
        #print(k)
        #print(x[k])
        try:
           # put the code you want to try here
           y[k] = math.gamma(x[k])
        except ValueError:
           # what to do if we get a value error
           y[k] = np.NaN
    
    plt.plot(x,y,"b")
    plt.hold('on')
    
    x = np.linspace(-4,-3,1000)
    x = x[1:-2]
    y = np.zeros((len(x),))

    for k in range(len(x)):
        #print(k)
        #print(x[k])
        try:
           # put the code you want to try here
           y[k] = math.gamma(x[k])
        except ValueError:
           # what to do if we get a value error
           y[k] = np.NaN
    
    plt.plot(x,y,"b")
     
    x = np.linspace(-3,-2,1000)
    x = x[1:-2]
    y = np.zeros((len(x),))

    for k in range(len(x)):
        #print(k)
        #print(x[k])
        try:
           # put the code you want to try here
           y[k] = math.gamma(x[k])
        except ValueError:
           # what to do if we get a value error
           y[k] = np.NaN
    
    plt.plot(x,y,"b")
    
    x = np.linspace(-2,-1,1000)
    x = x[1:-2]
    y = np.zeros((len(x),))

    for k in range(len(x)):
        #print(k)
        #print(x[k])
        try:
           # put the code you want to try here
           y[k] = math.gamma(x[k])
        except ValueError:
           # what to do if we get a value error
           y[k] = np.NaN
    
    plt.plot(x,y,"b")

    x = np.linspace(-1,0,1000)
    x = x[1:-2]
    y = np.zeros((len(x),))

    for k in range(len(x)):
        #print(k)
        #print(x[k])
        try:
           # put the code you want to try here
           y[k] = math.gamma(x[k])
        except ValueError:
           # what to do if we get a value error
           y[k] = np.NaN
    
    plt.plot(x,y,"b")

    x = np.linspace(0,5,1000)
    x = x[1:]
    y = np.zeros((len(x),))

    for k in range(len(x)):
        #print(k)
        #print(x[k])
        try:
           # put the code you want to try here
           y[k] = math.gamma(x[k])
        except ValueError:
           # what to do if we get a value error
           y[k] = np.NaN
    
    plt.plot(x,y,"b")
    
    plt.axvline(x=-4, color='r', linestyle='--')
    plt.axvline(x=-3, color='r', linestyle='--')
    plt.axvline(x=-2, color='r', linestyle='--')
    plt.axvline(x=-1, color='r', linestyle='--')
    plt.axvline(x=0, color='k')
    plt.axhline(y=0, color='k')
    plt.grid('on')
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\Gamma(x)$")
    plt.ylim([-5,5])      
    plt.show()

def plot_g(alpha, t, num):
    tau = np.linspace(0,t,num)
    plt.plot(tau,tau**alpha,"r")
    plt.plot(tau,(-1*tau)**alpha,"b")
    plt.plot(tau,(t-tau)**alpha,"m")
    plt.plot(tau,-1*(t-tau)**alpha,"c")
    plt.plot(tau,t**alpha-(t-tau)**alpha,"g")
    plt.show()

def plot_g_scaling(alpha=0.6,t=10,num_points=1000,k=3):
    
    tau = np.linspace(0,t*k,num_points)
    y = k**alpha*g_func(tau/k,alpha,t)
    y2 = g_func(tau,alpha,t*k)

    plt.plot(tau,y)
    plt.plot(tau,y2)

    plt.show()

    #tau = np.linspace(0,t,num_points)
    #tau2 = np.linspace(0,t*k,num_points)
    #y = g_func(tau,alpha,t)
    #y2 = g_func(k*tau,alpha,t*k)
    #plt.plot(tau,y)
    #plt.plot(tau,y*k**alpha)
    #plt.plot(tau,y2)
    #plt.show()   


def fp(x):
    return -1*(x*x) + 1 
    
def plot_inversepar(k=2):
    x = np.linspace(-1,1)
    x2 = x*k
    y = fp(x)
    y2 = k*fp(x2/k)
    plt.plot(x,y)
    plt.plot(x2,y2)

    plt.show()

def plot_y_test(alpha=0.5,t=10,num_points=200):
    y = np.linspace(0,t,num_points)
    tau = -1*g_func(y,alpha,t)
    plt.plot(tau,y)
    plt.plot(tau+y,y)
    plt.plot(y,y)
    plt.show()

def plot_frac_int1():
    t = np.linspace(0,10,1000)

    plt.plot(t,t,"k",lw=2,label=r"$\alpha=0$")
    y1 = (25.0/(6.0*math.gamma(1.0/5.0)))*t**(6.0/5.0)
    y2 = (25.0/(14.0*math.gamma(2.0/5.0)))*t**(7.0/5.0)
    y3 = (25.0/(24.0*math.gamma(3.0/5.0)))*t**(8.0/5.0)
    y4 = (25.0/(36.0*math.gamma(4.0/5.0)))*t**(9.0/5.0)
    plt.plot(t,y1,"k",dashes=[10, 5, 20, 5],label=r"$\alpha=0.2$")	
    plt.plot(t,y2,"k",dashes=[4,10],label=r"$\alpha=0.4$")
    plt.plot(t,y3,"k",ls=":",label=r"$\alpha=0.6$")
    plt.plot(t,y4,"k",ls="-.",label=r"$\alpha=0.8$")
    plt.plot(t,0.5*t*t,"k",dashes=[5,1],lw=2,label=r"$\alpha=1.0$")

    t_vector = np.array([2,4,6,8,10])

    v = np.linspace(0,1.5,len(t_vector))
    plt.legend()

    for k in range(len(v)):
    	y1 = (25.0/(6.0*math.gamma(1.0/5.0)))*t_vector[k]**(6.0/5.0)
    	y2 = (25.0/(14.0*math.gamma(2.0/5.0)))*t_vector[k]**(7.0/5.0)
    	y3 = (25.0/(24.0*math.gamma(3.0/5.0)))*t_vector[k]**(8.0/5.0)
    	y4 = (25.0/(36.0*math.gamma(4.0/5.0)))*t_vector[k]**(9.0/5.0)
    
    	plt.plot(t_vector[k],y1,"o",color=[v[k]/2,v[k]/2,v[k]/2])
    	plt.plot(t_vector[k],y2,"o",color=[v[k]/2,v[k]/2,v[k]/2])
    	plt.plot(t_vector[k],y3,"o",color=[v[k]/2,v[k]/2,v[k]/2])
    	plt.plot(t_vector[k],y4,"o",color=[v[k]/2,v[k]/2,v[k]/2])

    plt.xlim([0,10])
    plt.ylim([0,50])
    plt.xlabel(r"$t$")
    plt.ylabel(r"$y$")
    
    plt.show()

if __name__ == "__main__":
   plot_frac_int1()
   #plot_y_test()
   #plot_inversepar()
   #plt_gh_functions()
   #plot_b_functions()
   #plot_b_functions(f_inv = f2_inv, f=f2)
   #scaling() 
   #plot_g_scaling()
   #plot_g(1.5,10,1000)
   #x = np.linspace(-5,5,2000)
   #x = x[1:]
   #plot_gamma2(x)
   '''
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

