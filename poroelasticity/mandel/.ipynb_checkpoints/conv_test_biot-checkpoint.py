import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import scipy.optimize as opt
import porepy as pp
from porepy.ad.forward_mode import Ad_array
np.set_printoptions(precision=4, suppress = True)


def biot_convergence_in_space(N):    
    # coding: utf-8
    
    # ### Source terms and analytical solutions
    
    # In[330]:
    
    
    def source_flow(g,tau):
        
        x1 = g.cell_centers[0]
        x2 = g.cell_centers[1]
        
        f_flow = tau*(2*np.sin(2*np.pi*x2) - \
                 4*x1*np.pi**2*np.sin(2*np.pi*x2)*(x1 - 1)) - \
                 x1*np.sin(2*np.pi*x2) - \
                 np.sin(2*np.pi*x2)*(x1 - 1) + \
                 2*np.pi*np.cos(2*np.pi*x2)*np.sin(2*np.pi*x1) 
        
        return f_flow
    
    def source_mechanics(g):
        
        x1 = g.cell_centers[0]
        x2 = g.cell_centers[1]
        
        f_mech = np.zeros(g.num_cells * g.dim)
        
        f_mech[::2] = 6*np.sin(2*np.pi*x2) - \
                      x1*np.sin(2*np.pi*x2) -  \
                      np.sin(2*np.pi*x2)*(x1 - 1) - \
                      8*np.pi**2*np.cos(2*np.pi*x1)*np.cos(2*np.pi*x2) - \
                      4*x1*np.pi**2*np.sin(2*np.pi*x2)*(x1 - 1)
        
        f_mech[1::2] = 4*np.pi*np.cos(2*np.pi*x2)*(x1 - 1) + \
                       16*np.pi**2*np.sin(2*np.pi*x1)*np.sin(2*np.pi*x2) + \
                       4*x1*np.pi*np.cos(2*np.pi*x2) - \
                       2*x1*np.pi*np.cos(2*np.pi*x2)*(x1 - 1)
        
        return f_mech
    
    def analytical(g):
        
        sol = dict()
        x1 = g.cell_centers[0]
        x2 = g.cell_centers[1]
        
        sol['u'] = np.zeros(g.num_cells*g.dim)
        sol['u'][::2] = x1*(1-x1) * np.sin(2*np.pi*x2)
        sol['u'][1::2] = np.sin(2*np.pi*x1) * np.sin(2*np.pi*x2)
        
        sol['p'] = sol['u'][::2]
        
        return sol
    
    
    # ### Getting mechanics boundary conditions
    
    # In[331]:
    
    
    def get_bc_mechanics(g,b_faces,
                         x_min,x_max,west,east,
                         y_min,y_max,south,north):
            
        # Setting the tags at each boundary side for the mechanics problem
        labels_mech = np.array([None]*b_faces.size)
        labels_mech[west]   = 'dir' 
        labels_mech[east]   = 'dir'  
        labels_mech[south]  = 'dir'  
        labels_mech[north]  = 'dir' 
        
        # Constructing the bc object for the mechanics problem
        bc_mech = pp.BoundaryConditionVectorial(g, b_faces, labels_mech)
    
        # Constructing the boundary values array for the mechanics problem
        bc_val_mech = np.zeros(g.num_faces * g.dim)
        
        return bc_mech,bc_val_mech   
    
    
    # ### Getting flow boundary conditions
    
    # In[332]:
    
    
    def get_bc_flow(g,b_faces,
                    x_min,x_max,west,east,
                    y_min,y_max,south,north):
        
        # Setting the tags at each boundary side for the mechanics problem
        labels_flow = np.array([None]*b_faces.size)
        labels_flow[west]   = 'dir'     
        labels_flow[east]   = 'dir'     
        labels_flow[south]  = 'dir'    
        labels_flow[north]  = 'dir'   
    
        # Constructing the bc object for the flow problem
        bc_flow = pp.BoundaryCondition(g, b_faces, labels_flow)
    
        # Constructing the boundary values array for the flow problem
        bc_val_flow = np.zeros(g.num_faces)
        
        return bc_flow,bc_val_flow
    
    
    # ### Setting up the grid
    
    # In[333]:
    
    
    Nx = Ny = N
    Lx = 1; Ly = 1
    g = pp.CartGrid([Nx,Ny], [Lx,Ly])
    g.compute_geometry()
    V = g.cell_volumes
    
    
    # ### Physical parameters
    
    # In[334]:
    
    
    # Skeleton parameters
    mu_s = 1                                    # [Pa] Shear modulus
    lambda_s = 1                                # [Pa] Lame parameter
    K_s = (2/3) * mu_s + lambda_s               # [Pa] Bulk modulus
    E_s = mu_s * ((9*K_s)/(3*K_s+mu_s))         # [Pa] Young's modulus
    nu_s  = (3*K_s-2*mu_s)/(2*(3*K_s+mu_s))     # [-] Poisson's coefficient
    k_s = 1                                     # [m^2] Permeabiliy
    
    # Fluid parameters
    mu_f = 1                                    # [Pa s] Dynamic viscosity
    
    # Porous medium parameters
    alpha_biot = 1.                             # [m^2] Intrinsic permeability
    S_m = 0                                     # [1/Pa] Specific Storage
    
    
    # ### Creating second and fourth order tensors
    
    # In[335]:
    
    
    # Permeability tensor
    perm = pp.SecondOrderTensor(g.dim, 
                                k_s * np.ones(g.num_cells)) 
    
    # Stiffness matrix
    constit = pp.FourthOrderTensor(g.dim, 
                                   mu_s * np.ones(g.num_cells), 
                                   lambda_s * np.ones(g.num_cells))
    
    
    # ### Time parameters
    
    # In[336]:
    
    
    t0 = 0                                # [s] Initial time
    tf = 1                                # [s] Final simulation time
    tLevels = 1                           # [-] Time levels
    times = np.linspace(t0,tf,tLevels+1)  # [s] Vector of time evaluations
    dt = np.diff(times)                   # [s] Vector of time steps
    
    
    # ### Boundary conditions pre-processing
    
    # In[337]:
    
    
    b_faces = g.tags['domain_boundary_faces'].nonzero()[0]
    
    # Extracting indices of boundary faces w.r.t g
    x_min = b_faces[g.face_centers[0,b_faces] < 0.0001]
    x_max = b_faces[g.face_centers[0,b_faces] > 0.9999*Lx]
    y_min = b_faces[g.face_centers[1,b_faces] < 0.0001]
    y_max = b_faces[g.face_centers[1,b_faces] > 0.9999*Ly]
    
    # Extracting indices of boundary faces w.r.t b_faces
    west   = np.in1d(b_faces,x_min).nonzero()
    east   = np.in1d(b_faces,x_max).nonzero()
    south  = np.in1d(b_faces,y_min).nonzero()
    north  = np.in1d(b_faces,y_max).nonzero()
    
    # Mechanics boundary conditions
    bc_mech,bc_val_mech    = get_bc_mechanics(g,b_faces,
                                               x_min,x_max,west,east,
                                               y_min,y_max,south,north)   
    # FLOW BOUNDARY CONDITIONS
    bc_flow,bc_val_flow    = get_bc_flow(g,b_faces,
                                        x_min,x_max,west,east,
                                        y_min,y_max,south,north)
    
    
    # ### Initialiazing solution and solver dicitionaries
    
    # In[338]:
    
    
    # Solution dictionary
    sol = dict()
    sol['time'] = np.zeros(tLevels+1,dtype=float)
    sol['displacement'] = np.zeros((tLevels+1,g.num_cells*g.dim),dtype=float)
    sol['displacement_faces'] = np.zeros((tLevels+1,g.num_faces*g.dim*2),dtype=float)
    sol['pressure'] = np.zeros((tLevels+1,g.num_cells),dtype=float)
    sol['traction'] = np.zeros((tLevels+1,g.num_faces*g.dim),dtype=float)
    sol['flux'] = np.zeros((tLevels+1,g.num_faces),dtype=float)
    sol['iter'] = np.array([],dtype=int)
    sol['time_step'] = np.array([],dtype=float)
    sol['residual'] = np.array([],dtype=float)
    
    # Solver dictionary
    newton_param = dict()
    newton_param['tol'] = 1E-8       # maximum tolerance
    newton_param['max_iter'] = 20    # maximum number of iterations
    newton_param['res_norm'] = 1000  # initializing residual
    newton_param['iter'] = 1         # iteration
    
    
    # ### Discrete operators and discrete equations
    
    # ### Flow operators
    
    # In[339]:
    
    
    F       = lambda x: biot_F * x        # Flux operator
    boundF  = lambda x: biot_boundF * x   # Bound Flux operator
    compat  = lambda x: biot_compat * x   # Compatibility operator (Stabilization term)
    divF    = lambda x: biot_divF * x     # Scalar divergence operator
    
    
    # ### Mechanics operators
    
    # In[340]:
    
    
    S              = lambda x: biot_S * x                  # Stress operator
    boundS         = lambda x: biot_boundS * x             # Bound Stress operator
    divU           = lambda x: biot_divU * x               # Divergence of displacement field   
    divS           = lambda x: biot_divS * x               # Vector divergence operator
    gradP          = lambda x: biot_divS * biot_gradP * x  # Pressure gradient operator
    boundDivU      = lambda x: biot_boundDivU * x          # Bound Divergence of displacement operator
    boundUCell     = lambda x: biot_boundUCell * x         # Contribution of displacement at cells -> Face displacement
    boundUFace     = lambda x: biot_boundUFace * x         # Contribution of bc_mech at the boundaries -> Face displacement
    boundUPressure = lambda x: biot_boundUPressure * x     # Contribution of pressure at cells -> Face displacement
    
    
    # ### Discrete equations
    
    # In[341]:
    
    
    # Source terms
    f_mech = source_mechanics(g)
    f_flow = source_flow(g,dt[0])
    
    # Generalized Hooke's law
    T = lambda u: S(u) + boundS(bc_val_mech) 
    
    # Momentum conservation equation (I)
    u_eq1 = lambda u: divS(T(u)) 
    
    # Momentum conservation equation (II)
    u_eq2 = lambda p: -gradP(p) + f_mech * V[0]
    
    # Darcy's law
    Q = lambda p: (1./mu_f) * (F(p) + boundF(bc_val_flow))
    
    # Mass conservation equation (I)
    p_eq1 = lambda u,u_n: alpha_biot * divU(u-u_n)
    
    # Mass conservation equation (II)
    p_eq2 = lambda p,p_n,dt:  (p - p_n) * S_m * V  +                           divF(Q(p)) * dt +                           alpha_biot * compat(p - p_n) * V[0]-                           (f_flow/dt) * V[0]
    
    
    # ## Creating AD variables
    
    # In[343]:
    
    
    # Create displacement AD-variable
    u_ad = Ad_array(np.zeros(g.num_cells*2), sps.diags(np.ones(g.num_cells * g.dim)))
    
    # Create pressure AD-variable
    p_ad = Ad_array(np.zeros(g.num_cells), sps.diags(np.ones(g.num_cells)))
    
    
    # ## Performing discretization
    
    # In[344]:
    
    
    d = dict() # initialize dictionary to store data
    
    # Mechanics data object
    specified_parameters_mech = {"fourth_order_tensor": constit, 
                                 "bc": bc_mech, 
                                 "biot_alpha" : 1.,
                                 "bc_values": bc_val_mech}
    pp.initialize_default_data(g,d,"mechanics", specified_parameters_mech)
                                 
    # Flow data object
    specified_parameters_flow = {"second_order_tensor": perm, 
                                 "bc": bc_flow, 
                                 "biot_alpha": 1.,
                                 "bc_values": bc_val_flow}
    pp.initialize_default_data(g,d,"flow", specified_parameters_flow)
    
    
    # Biot discretization
    solver_biot = pp.Biot("mechanics","flow")
    solver_biot.discretize(g,d)
    
    # Mechanics discretization matrices
    biot_S = d['discretization_matrices']['mechanics']['stress']
    biot_boundS = d['discretization_matrices']['mechanics']['bound_stress']
    biot_divU = d['discretization_matrices']['mechanics']['div_d']
    biot_gradP = d['discretization_matrices']['mechanics']['grad_p']
    biot_boundDivU = d['discretization_matrices']['mechanics']['bound_div_d']
    biot_boundUCell = d['discretization_matrices']['mechanics']['bound_displacement_cell']
    biot_boundUFace = d['discretization_matrices']['mechanics']['bound_displacement_face']
    biot_boundUPressure = d['discretization_matrices']['mechanics']['bound_displacement_pressure']
    biot_divS = pp.fvutils.vector_divergence(g)
    
    # Flow discretization matrices
    biot_F = d['discretization_matrices']['flow']['flux']
    biot_boundF = d['discretization_matrices']['flow']['bound_flux']
    biot_compat = d['discretization_matrices']['flow']['biot_stabilization']
    biot_divF = pp.fvutils.scalar_divergence(g)
    
    # Saving initial condition
    sol['pressure'][0] = p_ad.val
    sol['displacement'][0] = u_ad.val
    sol['displacement_faces'][0] = ( boundUCell(sol['displacement'][0]) + 
                                      boundUFace(bc_val_mech) + 
                                      boundUPressure(sol['pressure'][0]))
    sol['time'][0] = times[0]
    sol['traction'][0] = T(u_ad.val)
    sol['flux'][0] = Q(p_ad.val)    
                           
    
    
    # ## The time loop
    
    # In[345]:
    
    
    tt = 0 # time counter
    
    while times[tt] < times[-1]:      
        
        tt += 1 # increasing time counter
          
        # Displacement and pressure at the previous time step
        u_n = u_ad.val.copy()          
        p_n = p_ad.val.copy()   
        
        # Updating residual and iteration at each time step
        newton_param.update({'res_norm':1000, 'iter':1}) 
        
        # Newton loop
        while newton_param['res_norm'] > newton_param['tol'] and newton_param['iter'] <= newton_param['max_iter']:
            
            # Calling equations
            eq1 = u_eq1(u_ad)
            eq2 = u_eq2(p_ad)
            eq3 = p_eq1(u_ad,u_n)
            eq4 = p_eq2(p_ad,p_n,dt[tt-1])
            
            # Assembling Jacobian of the coupled system
            J_mech = np.hstack((eq1.jac,eq2.jac)) # Jacobian blocks (mechanics)
            J_flow = np.hstack((eq3.jac,eq4.jac)) # Jacobian blocks (flow)
            J = sps.bmat(np.vstack((J_mech,J_flow)),format='csc') # Jacobian (coupled)
    
            # Determining residual of the coupled system
            R_mech = eq1.val + eq2.val            # Residual (mechanics)
            R_flow = eq3.val + eq4.val            # Residual (flow)
            R = np.hstack((R_mech,R_flow))        # Residual (coupled)
    
            y = sps.linalg.spsolve(J,-R)                  # 
            u_ad.val = u_ad.val + y[:g.dim*g.num_cells]   # Newton update
            p_ad.val = p_ad.val + y[g.dim*g.num_cells:]   #
            
            newton_param['res_norm'] = np.linalg.norm(R)  # Updating residual
            
            if newton_param['res_norm'] <= newton_param['tol'] and newton_param['iter'] <= newton_param['max_iter']:
                print('Iter: {} \t Error: {:.8f} [m]'.format(newton_param['iter'],newton_param['res_norm']))
            elif newton_param['iter'] > newton_param['max_iter']:
                print('Error: Newton method did not converge!')
            else:
                newton_param['iter'] += 1
        
        # Saving variables
        sol['iter'] = np.concatenate((sol['iter'],np.array([newton_param['iter']])))
        sol['residual'] = np.concatenate((sol['residual'],np.array([newton_param['res_norm']])))
        sol['time_step'] = np.concatenate((sol['time_step'],dt))    
        sol['pressure'][tt] = p_ad.val
        sol['displacement'][tt] = u_ad.val
        sol['displacement_faces'][tt] = (boundUCell(sol['displacement'][tt]) + 
                                          boundUFace(bc_val_mech) + 
                                          boundUPressure(sol['pressure'][tt]))
        sol['time'][tt] = times[tt]
        sol['traction'][tt] = T(u_ad.val)
        sol['flux'][tt] = Q(p_ad.val)
    
        # Determining analytical solution
        sol_anal = analytical(g)
        
        # Determining norms
        p_norm = np.linalg.norm(sol_anal['p'] - sol['pressure'][-1])/(np.linalg.norm(sol['pressure'][-1]))
        u_mag_num = np.sqrt(sol['displacement'][-1][::2]**2 + sol['displacement'][-1][1::2]**2)
        u_mag_ana = np.sqrt(sol_anal['u'][::2]**2 + sol_anal['u'][1::2]**2)
        u_norm = np.linalg.norm(u_mag_ana - u_mag_num)/np.linalg.norm(u_mag_num)
        
        return p_norm,u_norm