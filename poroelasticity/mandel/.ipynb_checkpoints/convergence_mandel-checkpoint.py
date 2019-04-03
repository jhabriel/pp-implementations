# Importing modules
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import scipy.optimize as opt
import porepy as pp
from porepy.ad.forward_mode import Ad_array
np.set_printoptions(precision=4, suppress = True)

def conv_test(n):
 
    # Analytical solution
    def mandel_solution(g,Nx,Ny,times,F,B,nu_u,nu,c_f,mu_s):
         
        # Some needed parameters
        a = np.max(g.face_centers[0])        # a = Lx
        x_cntr = g.cell_centers[0][:Nx]      # [m] vector of x-centers
        y_cntr = g.cell_centers[1][::Nx]     # [m] vector of y-centers
    
        # Solutions to tan(x) - ((1-nu)/(nu_u-nu)) x = 0
        """
        This is somehow tricky, we have to solve the equation numerically in order to
        find all the positive solutions to the equation. Later we will use them to 
        compute the infinite sums. Experience has shown that 200 roots are more than enough to
        achieve accurate results. Note that we find the roots using the bisection method.
        """
        f      = lambda x: np.tan(x) - ((1-nu)/(nu_u-nu))*x # define the algebraic eq. as a lambda function
        n_series = 200           # number of roots
        a_n = np.zeros(n_series) # initializing roots array
        x0 = 0                   # initial point
        for i in range(0,len(a_n)):
            a_n[i] = opt.bisect( f,                           # function
                                 x0+np.pi/4,                  # left point 
                                 x0+np.pi/2-10000*2.2204e-16, # right point (a tiny bit less than pi/2)
                                 xtol=1e-30,                  # absolute tolerance
                                 rtol=1e-15                   # relative tolerance
                               )        
            x0 += np.pi # apply a phase change of pi to get the next root
    
        # Creating dictionary to store analytical solutions
        mandel_sol = dict()
        mandel_sol['p'] = np.zeros((len(times),len(x_cntr)))
        mandel_sol['u_x'] = np.zeros((len(times),len(x_cntr)))
        mandel_sol['u_y'] = np.zeros((len(times),len(y_cntr)))
        mandel_sol['sigma_yy'] = np.zeros((len(times),len(x_cntr)))
    
        # Terms needed to compute the solutions (these are constants)  
        p0 = (2 * F * B * (1 + nu_u)) / (3*a)
        ux0_1 = ((F*nu)/(2*mu_s*a))
        ux0_2 = -((F*nu_u)/(mu_s*a))
        ux0_3 = F/mu_s
        uy0_1 = (-F*(1-nu))/(2*mu_s*a)
        uy0_2 = (F*(1-nu_u)/(mu_s*a))
        sigma0_1 = -F/a
        sigma0_2 = (-2*F*B*(nu_u-nu)) / (a*(1-nu))
        sigma0_3 = (2*F)/a
        
        # Saving solutions for the initial conditions
        mandel_sol['p'][0] = ((F * B * (1 + nu_u)) / (3*a)) * np.ones(Nx)
        mandel_sol['u_x'][0] = (F * nu_u * x_cntr)/(2*mu_s*a)
        mandel_sol['u_y'][0] = ((-F*(1-nu_u))/(2*mu_s*a)) * y_cntr
    
        # Storing solutions for the subsequent time steps
        for ii in range(1,len(times)):
       
           # Analytical Pressures
            p_sum = 0 
            for n in range(len(a_n)):   
                p_sum += ( 
                        ((np.sin(a_n[n]))/(a_n[n] - (np.sin(a_n[n]) * np.cos(a_n[n])))) * 
                        (np.cos((a_n[n]*x_cntr)/a) - np.cos(a_n[n])) * 
                        np.exp((-(a_n[n]**2) * c_f * times[ii])/(a**2))
                   )
           
            mandel_sol['p'][ii] = p0 * p_sum
        
            # Analytical horizontal displacements
            ux_sum1 = 0
            ux_sum2 = 0
            for n in range(len(a_n)):      
                ux_sum1 += ( 
                        (np.sin(a_n[n])*np.cos(a_n[n]))/(a_n[n] - np.sin(a_n[n])*np.cos(a_n[n])) *        
                        np.exp((-(a_n[n]**2) * c_f * times[ii])/(a**2)) 
                           )    
                ux_sum2 += (
                       (np.cos(a_n[n])/(a_n[n] - (np.sin(a_n[n]) * np.cos(a_n[n])))) *
                       np.sin(a_n[n] * (x_cntr/a)) * 
                       np.exp((-(a_n[n]**2) * c_f * times[ii])/(a**2)) 
                           ) 
            mandel_sol['u_x'][ii] =  (ux0_1 + ux0_2*ux_sum1) * x_cntr + ux0_3 * ux_sum2
           
            # Analytical vertical displacements
            uy_sum = 0
            for n in range(len(a_n)):
                uy_sum += (
                       ((np.sin(a_n[n]) * np.cos(a_n[n]))/(a_n[n] - np.sin(a_n[n]) * np.cos(a_n[n]))) * 
                       np.exp((-(a_n[n]**2) * c_f * times[ii])/(a**2)) 
                      )
            mandel_sol['u_y'][ii] =  (uy0_1 + uy0_2*uy_sum) * y_cntr
       
            # Analitical vertical stress
            sigma_sum1 = 0
            sigma_sum2 = 0
            for n in range(len(a_n)):
                sigma_sum1 += (
                          ((np.sin(a_n[n]))/(a_n[n] - (np.sin(a_n[n]) * np.cos(a_n[n])))) * 
                           np.cos(a_n[n] * (x_cntr/a)) * 
                           np.exp((-(a_n[n]**2) * c_f * times[ii])/(a**2)) 
                         )
                sigma_sum2 += (
                          ((np.sin(a_n[n])*np.cos(a_n[n]))/(a_n[n] - np.sin(a_n[n])*np.cos(a_n[n]))) * 
                          np.exp((-(a_n[n]**2) * c_f * times[ii])/(a**2))                  
                         )
            mandel_sol['sigma_yy'][ii] =  (sigma0_1 + sigma0_2*sigma_sum1) + (sigma0_3 * sigma_sum2)
           
        return mandel_sol   
    
    
    # Computing the initial condition
    
    def get_mandel_init_cond(g,F,B,nu_u,mu_s):
        
        # Initialing pressure and displacement arrays
        p0 = np.zeros(g.num_cells)
        u0 = np.zeros(g.num_cells*2)
        
        # Some needed parameters
        a = np.max(g.face_centers[0])        # a = Lx
        
        p0 = ((F * B * (1 + nu_u)) / (3*a)) * np.ones(g.num_cells)
        u0[::2] = (F * nu_u * g.cell_centers[0])/(2*mu_s*a)
        u0[1::2] = ((-F*(1-nu_u))/(2*mu_s*a)) * g.cell_centers[1]        
        
        return p0,u0  
    
    
    # Getting the time-dependent boundary condition
    
    def get_mandel_bc(g,y_max,times,F,B,nu_u,nu,c_f,mu_s):
        
        # Initializing top boundary array
        u_top = np.zeros((len(times),len(y_max)))
           
        # Some needed parameters
        a = np.max(g.face_centers[0])        # a = Lx
        b = np.max(g.face_centers[1])        # b = Ly
        y_top = g.face_centers[1][y_max]     # [m] y-coordinates at the top boundary
        
        # Solutions to tan(x) - ((1-nu)/(nu_u-nu)) x = 0
        """
        This is somehow tricky, we have to solve the equation numerically in order to
        find all the positive solutions to the equation. Later we will use them to 
        compute the infinite sums. Experience has shown that 200 roots are more than enough to
        achieve accurate results. Note that we find the roots using the bisection method.
        """
        f      = lambda x: np.tan(x) - ((1-nu)/(nu_u-nu))*x # define the algebraic eq. as a lambda function
        n_series = 200           # number of roots
        a_n = np.zeros(n_series) # initializing roots array
        x0 = 0                   # initial point
        for i in range(0,len(a_n)):
            a_n[i] = opt.bisect( f,                           # function
                                 x0+np.pi/4,                  # left point 
                                 x0+np.pi/2-10000*2.2204e-16, # right point (a tiny bit less than pi/2)
                                 xtol=1e-30,                  # absolute tolerance
                                 rtol=1e-15                   # relative tolerance
                                )        
            x0 += np.pi # apply a phase change of pi to get the next root
             
        # Terms needed to compute the solutions (these are constants)  
        uy0_1 = (-F*(1-nu))/(2*mu_s*a)
        uy0_2 = (F*(1-nu_u)/(mu_s*a))
        
        # For the initial condition:  
        u_top[0] = ((-F*(1-nu_u))/(2*mu_s*a)) * b
        
        for i in range(1,len(times)):  
           # Analytical vertical displacements at the top boundary
            uy_sum = 0
            for n in range(len(a_n)):
                uy_sum += (
                           ((np.sin(a_n[n]) * np.cos(a_n[n]))/(a_n[n] - np.sin(a_n[n]) * np.cos(a_n[n]))) * 
                           np.exp((-(a_n[n]**2) * c_f * times[i])/(a**2)) 
                          )
           
            u_top[i] =  (uy0_1 + uy0_2*uy_sum) * y_top
        
        # Returning array of u_y at the top boundary
        return u_top 
    
    
    # Getting mechanics boundary conditions
    
    def get_bc_mechanics(g,u_top,times,b_faces,
                         x_min,x_max,west,east,
                         y_min,y_max,south,north):
            
        # Setting the tags at each boundary side for the mechanics problem
        labels_mech = np.array([None]*b_faces.size)
        labels_mech[west]   = 'dir_x' # roller
        labels_mech[east]   = 'neu'   # traction free 
        labels_mech[south]  = 'dir_y' # roller 
        labels_mech[north]  = 'dir_y' # roller (with non-zero displacement in the vertical direction)
    
        # Constructing the bc object for the mechanics problem
        bc_mech = pp.BoundaryConditionVectorial(g, b_faces, labels_mech)
    
        # Constructing the boundary values array for the mechanics problem
        bc_val_mech = np.zeros((len(times),g.num_faces * g.dim,))
    
        for i in range(len(times)):
        
            # West side boundary conditions (mech)
            bc_val_mech[i][2*x_min] = 0                    # [m]
            bc_val_mech[i][2*x_min+1] = 0                  # [Pa]
        
            # East side boundary conditions (mech)
            bc_val_mech[i][2*x_max] = 0                    # [Pa]
            bc_val_mech[i][2*x_max+1] = 0                  # [Pa]
        
            # South Side boundary conditions (mech)
            bc_val_mech[i][2*y_min] = 0                    # [Pa]
            bc_val_mech[i][2*y_min+1] = 0                  # [m]
        
            # North Side boundary conditions (mech)
            bc_val_mech[i][2*y_max] = 0                    # [Pa]
            bc_val_mech[i][2*y_max+1] =  u_top[i]          # [m]
        
        return bc_mech,bc_val_mech   
    
    
    # Getting flow boundary conditions
    
    def get_bc_flow(g,b_faces,
                    x_min,x_max,west,east,
                    y_min,y_max,south,north):
        
        # Setting the tags at each boundary side for the mechanics problem
        labels_flow = np.array([None]*b_faces.size)
        labels_flow[west]   = 'neu'     # no flow 
        labels_flow[east]   = 'dir'     # constant pressure
        labels_flow[south]  = 'neu'     # no flow
        labels_flow[north]  = 'neu'     # no flow
    
        # Constructing the bc object for the flow problem
        bc_flow = pp.BoundaryCondition(g, b_faces, labels_flow)
    
        # Constructing the boundary values array for the flow problem
        bc_val_flow = np.zeros(g.num_faces)
        
        # West side boundary condition (flow)
        bc_val_flow[x_min] = 0                      # [Pa]
    
        # East side boundary condition (flow)
        bc_val_flow[x_max] = 0                      # [m^3/s]
    
        # South side boundary condition (flow)   
        bc_val_flow[y_min] = 0                      # [m^3/s]
    
        # North side boundary condition (flow)
        bc_val_flow[y_max] = 0                      # [m^3/s]    
        
        return bc_flow,bc_val_flow


    # ## Setting up the grid
    
    # In[7]:
    
    
    Nx = 40; Ny = 40
    Lx = 100; Ly = 10
    g = pp.CartGrid([Nx,Ny], [Lx,Ly])
    #g.nodes = g.nodes + 1E-7*np.random.randn(3, g.num_nodes)
    g.compute_geometry()
    V = g.cell_volumes
    
    # Physical parameters
    
    # Skeleton parameters
    mu_s = 2.475E+09                                                             # [Pa] Shear modulus
    lambda_s = 1.65E+09                                                          # [Pa] Lame parameter
    K_s = (2/3) * mu_s + lambda_s                                                # [Pa] Bulk modulus
    E_s = mu_s * ((9*K_s)/(3*K_s+mu_s))                                          # [Pa] Young's modulus
    nu_s  = (3*K_s-2*mu_s)/(2*(3*K_s+mu_s))                                      # [-] Poisson's coefficient
    k_s = 100 * 9.869233E-13                                                     # [m^2] Permeabiliy
    
    # Fluid parameters
    mu_f = 10.0E-3                                                               # [Pa s] Dynamic viscosity
    
    # Porous medium parameters
    alpha_biot = 1.                                                              # [m^2] Intrinsic permeability
    S_m = 6.0606E-11                                                             # [1/Pa] Specific Storage
    K_u = K_s + (alpha_biot**2)/S_m                                              # [Pa] Undrained bulk modulus
    B = alpha_biot / (S_m * K_u)                                                 # [-] Skempton's coefficient
    nu_u = (3*nu_s + B*(1-2*nu_s))/(3-B*(1-2*nu_s))                              # [-] Undrained Poisson's ratio
    c_f = (2*k_s*(B**2)*mu_s*(1-nu_s)*(1+nu_u)**2)/(9*mu_f*(1-nu_u)*(nu_u-nu_s)) # [m^2/s] Fluid diffusivity
        
    # Creating second and fourth order tensors
    
    # Permeability tensor
    perm = pp.SecondOrderTensor( g.dim, 
                                 k_s * np.ones(g.num_cells)
                               ) 
    # Stiffness matrix
    constit = pp.FourthOrderTensor( g.dim, 
                                    mu_s * np.ones(g.num_cells), 
                                    lambda_s * np.ones(g.num_cells)
                                  )
    # Time parameters    
    
    t0 = 0                                # [s] Initial time
    tf = 100                              # [s] Final simulation time
    tLevels = 100                         # [-] Time levels
    times = np.linspace(t0,tf,tLevels+1)  # [s] Vector of time evaluations
    dt = np.diff(times)                   # [s] Vector of time steps
    
    
    # Boundary conditions pre-processing    
    
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
    
    # Applied load and top boundary condition
    F_load = 6.8E+8                                                      # [N/m] Applied load
    u_top = get_mandel_bc(g,y_max,times,F_load,B,nu_u,nu_s,c_f,mu_s)     # [m] Vector of imposed vertical displacements
    
    # MECHANICS BOUNDARY CONDITIONS
    bc_mech,bc_val_mech    = get_bc_mechanics(g,u_top,times,b_faces,
                                               x_min,x_max,west,east,
                                               y_min,y_max,south,north)   
    # FLOW BOUNDARY CONDITIONS
    bc_flow,bc_val_flow    = get_bc_flow(g,b_faces,
                                        x_min,x_max,west,east,
                                        y_min,y_max,south,north)
    
    
    # Initialiazing solution and solver dicitionaries    
    
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
    newton_param['tol'] = 1E-6       # maximum tolerance
    newton_param['max_iter'] = 20    # maximum number of iterations
    newton_param['res_norm'] = 1000  # initializing residual
    newton_param['iter'] = 1         # iteration
    
    
    # Discrete operators and discrete equations
    
    # Flow operators    
    
    F       = lambda x: biot_F * x        # Flux operator
    boundF  = lambda x: biot_boundF * x   # Bound Flux operator
    compat  = lambda x: biot_compat * x   # Compatibility operator (Stabilization term)
    divF    = lambda x: biot_divF * x     # Scalar divergence operator
    
    
    # Mechanics operators    
    
    S              = lambda x: biot_S * x                  # Stress operator
    boundS         = lambda x: biot_boundS * x             # Bound Stress operator
    divU           = lambda x: biot_divU * x               # Divergence of displacement field   
    divS           = lambda x: biot_divS * x               # Vector divergence operator
    gradP          = lambda x: biot_divS * biot_gradP * x  # Pressure gradient operator
    boundDivU      = lambda x: biot_boundDivU * x          # Bound Divergence of displacement operator
    boundUCell     = lambda x: biot_boundUCell * x         # Contribution of displacement at cells -> Face displacement
    boundUFace     = lambda x: biot_boundUFace * x         # Contribution of bc_mech at the boundaries -> Face displacement
    boundUPressure = lambda x: biot_boundUPressure * x     # Contribution of pressure at cells -> Face displacement
    
    
    # Discrete equations    
    
    # Generalized Hooke's law
    T = lambda u,bc_val_mech: S(u) + boundS(bc_val_mech)
    
    # Momentum conservation equation (I)
    u_eq1 = lambda u,bc_val_mech: divS(T(u,bc_val_mech))
    
    # Momentum conservation equation (II)
    u_eq2 = lambda p: - gradP(p)
    
    # Darcy's law
    Q = lambda p: (1./mu_f) * (F(p) + boundF(bc_val_flow))
    
    # Mass conservation equation (I)
    p_eq1 = lambda u,u_n,bc_val_mech,bc_val_mech_n: alpha_biot * (divU(u-u_n) + boundDivU(bc_val_mech - bc_val_mech_n))
    
    # Mass conservation equation (II)
    p_eq2 = lambda p,p_n,dt:  (p - p_n) * S_m * V  +                           divF(Q(p)) * dt +                           alpha_biot * compat(p - p_n)
    
    
    # Creating AD variables    
    
    # Retrieve initial conditions 
    p_init,u_init = get_mandel_init_cond(g,F_load,B,nu_u,mu_s)
    
    # Create displacement AD-variable
    u_ad = Ad_array(u_init.copy(), sps.diags(np.ones(g.num_cells * g.dim)))
    
    # Create pressure AD-variable
    p_ad = Ad_array(p_init.copy(), sps.diags(np.ones(g.num_cells)))
    
    
    # The time loop    
    
    tt = 0 # time counter
    
    while times[tt] < times[-1]:
        
        ################################
        # Initializing data dictionary #
        ################################
        
        d = dict() # initialize dictionary to store data
        
        ################################
        #  Creating the data objects   #
        ################################
        
        # Mechanics data object
        specified_parameters_mech = {"fourth_order_tensor": constit, 
                                 "bc": bc_mech, 
                                 "biot_alpha" : 1.,
                                 "bc_values": bc_val_mech[tt], 
                                 "mass_weight":S_m}
    
        pp.initialize_default_data(g,d,"mechanics", specified_parameters_mech)
    
        # Flow data object
        specified_parameters_flow = {"second_order_tensor": perm, 
                                 "bc": bc_flow, 
                                 "biot_alpha": 1.,
                                 "bc_values": bc_val_flow, 
                                 "mass_weight": S_m, 
                                 "time_step":dt[tt-1]}
    
        pp.initialize_default_data(g,d,"flow", specified_parameters_flow)
        
        
        ################################
        #  CALLING MPFA/MPSA ROUTINES  #
        ################################
    
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
        
        ################################
        #  Saving Initial Condition    #
        ################################
        
        if times[tt] == 0:
            sol['pressure'][tt] = p_ad.val
            sol['displacement'][tt] = u_ad.val
            sol['displacement_faces'][tt] = ( 
                                              boundUCell(sol['displacement'][tt]) + 
                                              boundUFace(bc_val_mech[tt]) + 
                                              boundUPressure(sol['pressure'][tt])
                                            )
            sol['time'][tt] = times[tt]
            sol['traction'][tt] = T(u_ad.val,bc_val_mech[tt])
            sol['flux'][tt] = Q(p_ad.val)  
        
        tt += 1 # increasing time counter
          
        ################################
        #  Solving the set of PDE's    #
        ################################
        
        # Displacement and pressure at the previous time step
        u_n = u_ad.val.copy()          
        p_n = p_ad.val.copy()   
        
        # Updating residual and iteration at each time step
        newton_param.update({'res_norm':1000, 'iter':1}) 
        
        # Newton loop
        while newton_param['res_norm'] > newton_param['tol'] and newton_param['iter'] <= newton_param['max_iter']:
            
            # Calling equations
            eq1 = u_eq1(u_ad,bc_val_mech[tt])
            eq2 = u_eq2(p_ad)
            eq3 = p_eq1(u_ad,u_n,bc_val_mech[tt],bc_val_mech[tt-1])
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
        
        ################################
        #      Saving the variables    #
        ################################
        sol['iter'] = np.concatenate((sol['iter'],np.array([newton_param['iter']])))
        sol['residual'] = np.concatenate((sol['residual'],np.array([newton_param['res_norm']])))
        sol['time_step'] = np.concatenate((sol['time_step'],dt))    
        sol['pressure'][tt] = p_ad.val
        sol['displacement'][tt] = u_ad.val
        sol['displacement_faces'][tt] = (
                                          boundUCell(sol['displacement'][tt]) + 
                                          boundUFace(bc_val_mech[tt]) + 
                                          boundUPressure(sol['pressure'][tt])
                                        )
        sol['time'][tt] = times[tt]
        sol['traction'][tt] = T(u_ad.val,bc_val_mech[tt])
        sol['flux'][tt] = Q(p_ad.val)
       
    # Calling analytical solution    
    
    sol_mandel = mandel_solution( 
                                  g,Nx,Ny,times, 
                                  F_load,B,nu_u,
                                  nu_s,c_f,mu_s
                                )
    
    # Creating analytical and numerical results arrays
    
    p_num = (Lx*sol['pressure'][-1][:Nx])/F
    p_ana = (Lx*sol_mandel['p'][-1])/F
    
    # Returning values
    
    return p_num,p_ana

