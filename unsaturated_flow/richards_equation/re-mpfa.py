#%% Importing modules and external functions

import numpy as np
import scipy.sparse as sps
import porepy as pp
from porepy.ad.forward_mode import Ad_array
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress = True)

from functions_re import arithmetic_mpfa_hyd
from functions_re import time_stepping
from functions_re import newton_solver
from functions_re import save_solution
     
#%% Setting up the grid

Nx = 5;  Ny = 5;  Nz = 30
Lx = 10; Ly = 10; Lz = 100
g = pp.CartGrid([Nx,Ny,Nz], [Lx,Ly,Lz])
g.compute_geometry()
V = g.cell_volumes

#%% Fluid properties

fluid = dict()
fluid['density'] = 1.0       # [g/cm^3]
fluid['viscosity'] = 0.01    # [g/cm.s]
fluid['gravity'] = 980.6650  # [cm/s^2]

#%% Porous medium properties

medium = dict()
medium['sat_hyd_cond'] = 0.00922 # Saturated hydraulic conductivity [cm/s]

#%% Rock properties

rock = dict()
rock['permeability'] = (medium['sat_hyd_cond']*fluid['viscosity']) / \
                       (fluid['density']*fluid['gravity']) # [cm^2]

# Creating the porepy permeability object                       
perm = pp.SecondOrderTensor(g.dim, rock['permeability'] * np.ones(g.num_cells))

#%% van Genuchten parameters

van_gen = dict()
van_gen['alpha'] = 0.0335            # [1/cm] equation parameter
van_gen['n'] = 2.0                   # [-] equation parameter
van_gen['m'] = 1-(1/van_gen['n'])    # [-] equation parameter
van_gen['theta_s'] = 0.368           # [-] water content at saturated conditions
van_gen['theta_r'] = 0.102           # [-] residual water content

#%% Boundary and initial conditions

b_faces = g.tags['domain_boundary_faces'].nonzero()[0]

# Extracting grid information
z_cntr = g.cell_centers[2,:]
z_fcs = g.face_centers[2,:]

# Extracting indices of boundary faces w.r.t g
x_min = b_faces[g.face_centers[0,b_faces] < 0.0001]
x_max = b_faces[g.face_centers[0,b_faces] > 0.9999*Lx]
y_min = b_faces[g.face_centers[1,b_faces] < 0.0001]
y_max = b_faces[g.face_centers[1,b_faces] > 0.9999*Ly]
z_min = b_faces[g.face_centers[2,b_faces] < 0.0001]
z_max = b_faces[g.face_centers[2,b_faces] > 0.9999*Lz]

# Extracting indices of boundary faces w.r.t b_faces
east   = np.in1d(b_faces,x_min).nonzero()
west   = np.in1d(b_faces,x_max).nonzero()
south  = np.in1d(b_faces,y_min).nonzero()
north  = np.in1d(b_faces,y_max).nonzero()
bottom = np.in1d(b_faces,z_min).nonzero()
top    = np.in1d(b_faces,z_max).nonzero()

# Setting the tags at each boundary side
labels = np.array([None]*b_faces.size)
labels[east]   = 'neu'
labels[west]   = 'neu'
labels[south]  = 'neu'
labels[north]  = 'neu'
labels[bottom] = 'dir'
labels[top]    = 'dir'

# Constructing the bc object
bc = pp.BoundaryCondition(g, b_faces, labels)

# Imposed pressures and fluxes at the boundaries
flux_west = 0       # [cm^3/s] 
fluw_east = 0       # [cm^3/s]
flux_south = 0      # [cm^3/s]
flux_north = 0      # [cm^3/s]
psi_top = -75       # [cm]
psi_bottom = -1000  # [cm]

h_top = psi_top + z_fcs[z_max]       # [cm] top hyd head
h_bottom = psi_bottom + z_fcs[z_min] # [cm] bottom hyd head

# Constructing the boundary values array
bc_val = np.zeros(g.num_faces)
bc_val[x_min] = np.zeros(x_min.size)
bc_val[x_max] = np.zeros(x_max.size)
bc_val[y_min] = np.zeros(y_min.size)
bc_val[y_max] = np.zeros(y_max.size)
bc_val[z_min] = h_bottom * np.ones(z_min.size)
bc_val[z_max] = h_top * np.ones(z_max.size)

# Initial condition
h_init = psi_bottom + z_cntr

#%% Creating the data object

specified_parameters = {"second_order_tensor": perm, 
                        "bc": bc, 
                        "bc_values": bc_val}

data = pp.initialize_default_data(g,{},"flow", specified_parameters)

#%% Calling the mpfa routine

# Instatiating the pp.Mpfa class
solver = pp.Mpfa("flow")

# MPFA discretization
mpfa_F, mpfa_boundF, _, _ = solver.mpfa(g, perm, bc)
mpfa_div = pp.fvutils.scalar_divergence(g)

#%% Creating mpfa operators

F      = lambda x: mpfa_F * x
boundF = lambda x: mpfa_boundF * x
div    = lambda x: mpfa_div * x

#%% Water retention curves

"""
Please note that psi must be passed in centimeters!
"""

# Boolean lamdba-function which determines sat or unsat condition
is_unsat = lambda psi: psi < 0

# Water content
theta = lambda psi: is_unsat(psi) * ( (van_gen['theta_s'] - van_gen['theta_r']) / 
        (1. + (van_gen['alpha'] * np.abs(psi))**van_gen['n'])**van_gen['m'] + 
        van_gen['theta_r']) + (1-is_unsat(psi))*van_gen['theta_s']

# Specific moisture capacity
C = lambda psi: is_unsat(psi) * ( (van_gen['m'] * van_gen['n'] * psi * 
           (van_gen['theta_r']-van_gen['theta_s']) * 
            van_gen['alpha']**van_gen['n'] * np.abs(psi)**(van_gen['n']-2.) ) /
           (van_gen['alpha']**van_gen['n'] * np.abs(psi)**van_gen['n'] + 1.) ** 
           (van_gen['m']+1.)) + (1-is_unsat(psi)) * 0

# Relative permeability
krw = lambda psi: is_unsat(psi)*((1. - (van_gen['alpha'] * np.abs(psi)) **
            (van_gen['n']-1) * (1. + (van_gen['alpha'] * np.abs(psi)) ** 
             van_gen['n'])**(-van_gen['m']))**2. /
            (1. + (van_gen['alpha'] * np.abs(psi)) ** van_gen['n']) ** 
            (van_gen['m']/2.)) + (1-is_unsat(psi)) * 1.
                
#%% Discrete equations

"""
When we perform arithmetic operations involving an nd.array and an AD Porepy 
object, the array must be always located at the right of the AD Porepy object.
This behaviour has its root on the way NumPy perform multiplication
between objects of different size, the so call broadcasting property. 
Alternatively, we could create instead SciPy diagonal (sparse) matrices. 
However, what we can win from keeping the original structure of the equations,
is completely lost from the lack of readibility.
Let keep it this way until a major change is needed 
"""

# Arithmetic mean
krw_ar = lambda h_m: arithmetic_mpfa_hyd(krw,g,bc,bc_val,h_m)

# Multiphase Darcy Flux
q = lambda h,h_m: (fluid['density']*fluid['gravity']/fluid['viscosity']) * \
                   (F(h) + boundF(bc_val)) * krw_ar(h_m)

# Mass Conservation
h_eq = lambda h,h_n,h_m,dt: div(q(h,h_m)) +  (   (h-h_m)*C(h_m-z_cntr) + 
                                                    theta(h_m-z_cntr) - 
                                                    theta(h_n-z_cntr)
                                                 ) * (V/dt)                

#%% Creating the AD-object

h_ad = Ad_array(h_init.copy(), sps.diags(np.ones(g.num_cells)))

#%% Time, solver, printing and solution dictionaries

# Time parameters
time_param = dict()                       # initializing time parameters dictionary
time_param['sim_time'] = 72 * 3600        # [s] final simulation time
time_param['dt_init'] = 100               # [s] initial time step
time_param['dt_min'] = 100                # [s] minimum time step
time_param['dt_max'] = 10000              # [s] maximum time step
time_param['lower_opt_iter_range'] = 3    # [iter] lower optimal iteration range
time_param['upper_opt_iter_range'] = 7    # [iter] upper optimal iteration range
time_param['lower_mult_factor'] = 1.3     # [-] lower multiplication factor
time_param['upper_mult_factor'] = 0.7     # [-] upper multiplication factor
time_param['dt'] = time_param['dt_init']  # [s] initializing time step
time_param['dt_print'] = time_param['dt'] # [s] initializing printing time step
time_param['time_cum'] = 0                # [s] cumulative time

# Newton parameters
newton_param = dict()
newton_param['max_tol'] = 1               # [cm] maximum absolute tolerance (pressure head)
newton_param['max_iter'] = 10             # [iter] maximum number of iterations
newton_param['abs_tol'] = 100             # [cm] absolute tolerance
newton_param['iter'] = 1                  # [iter] iteration

# Printing parameters
print_param = dict()
print_param['levels'] = 10                # number of printing levels
print_param['times'] = np.linspace(time_param['sim_time']/print_param['levels'], \
                       time_param['sim_time'],print_param['levels'])
print_param['counter'] = 0                # initializing printing counter
print_param['is_active'] = False          # Printing = True; Not Printing = False

# Solution dictionary
sol = dict()
sol['time'] = np.zeros((print_param['levels']+1,1))
sol['hydraulic_head'] = np.zeros((print_param['levels']+1,g.num_cells))
sol['pressure_head'] = np.zeros((print_param['levels']+1,g.num_cells))
sol['elevation_head'] = np.zeros((print_param['levels']+1,g.num_cells))
sol['water_content'] = np.zeros((print_param['levels']+1,g.num_cells))
sol['darcy_fluxes'] = np.zeros((print_param['levels']+1,g.num_faces))
sol['iterations'] = np.array([],dtype=int)
sol['time_step'] = np.array([],dtype=float)
# Saving inital conditions
save_solution(sol,newton_param,time_param,print_param,g,h_ad,h_ad,theta,q)

#%% Time loop

while time_param['time_cum'] < time_param['sim_time']:
        
    if print_param['is_active'] == False:
        
        h_n = h_ad.val.copy()                                     # current time level (n)
        time_param['time_cum'] += time_param['dt'] 
        newton_param.update({'abs_tol':100, 'iter':1})            # updating tolerance and iterations
        
        # Newton loop
        while newton_param['abs_tol'] > newton_param['max_tol']   and \
              newton_param['iter']    < newton_param['max_iter']:      
              h_m = h_ad.val.copy()                               # current iteration level (m)
              eq = h_eq(h_ad,h_n,h_m,time_param['dt'])            # calling discrete equation
              newton_solver(eq,h_ad,h_m,newton_param,time_param,print_param)  # calling newton solver
        
        # Calling time stepping routine
        time_stepping(time_param,newton_param,print_param)
        
        # Determining if next step we should print or not
        if time_param['dt'] + time_param['time_cum'] >= print_param['times'][print_param['counter']]:
            time_param['dt_print'] = print_param['times'][print_param['counter']] - time_param['time_cum']
            print_param['is_active'] = True
        
    elif print_param['is_active'] == True:
        
        h_ad_print = Ad_array(h_ad.val.copy(), sps.diags(np.ones(g.num_cells)))
        h_n_print = h_ad_print.val.copy()
        newton_param.update({'abs_tol':100, 'iter':1})            # updating tolerance and iterations
        
        # Newton loop
        while newton_param['abs_tol'] > newton_param['max_tol']   and \
              newton_param['iter']    < newton_param['max_iter']:      
              h_m_print = h_ad_print.val.copy()                                           # current iteration level (m)
              eq = h_eq(h_ad_print,h_n_print,h_m_print,time_param['dt_print'])            # calling discrete equation
              newton_solver(eq,h_ad_print,h_m_print,newton_param,time_param,print_param)  # calling newton solver
        
        print_param['is_active'] = False                          # Deactivating printing mode
        
        # Determining if we reach the end of the simulation or not
        if time_param['time_cum'] + time_param['dt_print'] == time_param['sim_time']:
            print('---- End of simulation ----')
            break
        
    # Saving solutions
    save_solution(sol,newton_param,time_param,print_param,g,h_ad,h_m,theta,q)
    
    
#%% Plotting results
fig = plt.figure(figsize=(15,12))

ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

for graph in range(len(sol['time'])):
    
    ax1.plot(sol['pressure_head'][graph],z_cntr,
             marker = '*', linewidth = 2,
             label=str(sol['time'][graph][0]/3600) + ' [h]')
    
    ax2.plot(sol['water_content'][graph],z_cntr,
             marker = '*', linewidth = 2,
             label=str(sol['time'][graph][0]/3600) + ' [h]')
    
ax3.plot(sol['iterations'],marker='o',linewidth=0.3)    
ax4.plot(sol['time_step'],marker='o',linewidth=0.3)

# Shrink current axis by 20%
box1 = ax1.get_position()
box2 = ax2.get_position()
box3 = ax3.get_position()
box4 = ax4.get_position()
ax1.set_position([box1.x0, box1.y0, box1.width * 0.7, box1.height])
ax2.set_position([box2.x0, box2.y0, box2.width * 0.7, box2.height])
ax3.set_position([box3.x0, box3.y0*0.7, box3.width * 0.7, box3.height])
ax4.set_position([box4.x0, box4.y0*0.7, box4.width * 0.7, box4.height])


# Put a legend to the right of the current axis
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Put axis labels
ax1.set_xlabel('Pressure Head [cm]')
ax1.set_ylabel('Elevation Head [cm]')
ax1.grid(True)  

ax2.set_xlabel('Hydraulic Head [cm]')
ax2.set_ylabel('Elevation Head [cm]')
ax2.grid(True)  

ax3.set_xlabel('Time Level')
ax3.set_ylabel('Number of iterations')
ax3.set_yticks([0,1,2,3,4,5,6])
ax3.grid(True)  

ax4.set_xlabel('Time Level')
ax4.set_ylabel('Time Step [s]')
ax4.grid(True)  

plt.show()

#%% Exporting to paraview

save = pp.viz.exporter.Exporter(g,"solution","viz")

for i in range(len(sol['time'])):
    save.write_vtk({"Pressure Head [cm]": sol['pressure_head'][i], 
                    "Water Content [-]": sol['water_content'][i],
                    "Water Saturation [-]": sol['water_content'][i]/van_gen['theta_s']},
                    time_step = i)
save.write_pvd(sol['time'])




        