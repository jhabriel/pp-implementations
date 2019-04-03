
#%% Importing modules and functions
import numpy as np
import scipy.sparse as sps
import porepy as pp
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from porepy.ad.forward_mode import Ad_array
from biot_convergence import biot_convergence
np.set_printoptions(precision=4, suppress = True)

#%% Analysis

n_cells = np.array([8,16,32,64])
t_steps = np.array([1,0.5,0.25,0.125])
h = 1/n_cells

error_values = np.zeros((len(n_cells),len(t_steps)))

for i in range(len(n_cells)):
    for j in range(len(t_steps)):
        error_values[i,j] = biot_convergence(n_cells[i],t_steps[j])
        
#%% Applying least squares for fitting a plane
        
h_v = np.log(np.matlib.repeat(h,len(t_steps)))
t_v = np.log(np.matlib.repmat(t_steps,1,np.size(h)))  
z = np.log(np.matlib.reshape(error_values,np.size(t_v)))


A = np.transpose(np.vstack((h_v,t_v,np.ones(np.size(t_v)))))
b = np.reshape(z,(np.size(t_v),1))

C = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(A),A)),np.transpose(A)),b)        

#%% Error reduction
error_red_h = np.zeros((len(h)-1,len(t_steps)))
for i in range(len(t_steps)-1):
    for j in range(len(t_steps)):
        error_red_h[i,j] = error_values[i,j]/error_values[i+1,j]
        
error_red_t = np.zeros((len(t_steps),len(h)-1))
for i in range(len(t_steps)):
    for j in range(len(h)-1):
        error_red_t[i,j] = error_values[i,j]/error_values[i,j+1]        
    
#%%
# Straight line
def func(x,A,B): # this is your 'straight line' y=f(x),
    return A*x + B

# Creating figure object
fig = plt.figure(figsize=(8,8))

ax1 = plt.subplot(111)
ax1.loglog(h,error_values,'-o',linewidth=2,label = 'Numerical data')
ax1.grid(True)
ax1.set_xlabel(r'$h$', fontsize=14)
ax1.set_ylabel(r'$\epsilon_p$',fontsize=14)

for i in range(len(t_steps)):
    popt,pcov = opt.curve_fit(func, np.log(h), np.log(error_values[:,i])) # your data x, y to fit
    A = popt[0]
    B = popt[1]
    print(A,B)

for i in range(len(h)):
    popt,pcov = opt.curve_fit(func, np.log(t_steps), np.log(error_values[i,:])) # your data x, y to fit
    A = popt[0]
    B = popt[1]
    print(A,B)
    
residuals = np.log(error_values) - func(np.log(h), A,B)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((np.log(error_values)-np.mean(np.log(error_values)))**2)
r_squared = 1 - (ss_res / ss_tot)
x_fit = np.array([np.min(np.log(h)),np.max(np.log(h))])
y_fit = func(x_fit,A,B)
ax1.loglog(np.exp(x_fit),np.exp(y_fit),'-k',linewidth=2,label='Fitted line')
ax1.text(0.2,0.8,r'$R^2$= {:.6}'.format(r_squared),size=15,transform=ax1.transAxes)
ax1.text(0.2,0.75,r'$p$={:.4}'.format(A),size=15,transform=ax1.transAxes)
plt.title(r'Convergence in space for $\tau$ = 1 [s]', size=16)
ax1.legend(loc='lower right',fontsize=15)
plt.show()

#%%
# Fitting a plane
def f_plane(x0,x1,a0,a1,a2):
    return a0 + a1*x0 + a2*x1

X,Y = np.meshgrid(h,t_steps)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(h_v,t_v,b)
ax.scatter(h_v,t_v,f_plane(h_v,t_v,C[0],C[1],C[2]))

ax.set_xlabel(r'$\log (h)$')
ax.set_ylabel(r'$\log (\Delta t)$')
ax.set_zlabel(r'$\log (\epsilon)_p$')


plt.show()






