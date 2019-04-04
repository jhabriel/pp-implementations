#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 21:13:33 2019

@author: jv
"""

#%% Importing modules
import numpy as np
import scipy.sparse as sps

#%% Artihmetic average

def arithmetic_mpfa_hyd(krw,g,bc,bc_val,h_m0):
    
    """
    Computes the arithmetic average of the relative permability
    
    SYNOPSIS:
        arithmetic_mpfa_hyd(krw,g,bc_val,h_m0)
        
    INPUT ARGUMENTS:
        krw         - Lambda function, relative permeability function krw = f(psi)
        g           - PorePy grid object
        bc_val      - NumPy array, containing values of boundary conditions
        h_m0        - NumPy array, containing values of hydraulic head at the cell centers
        
    RETURNS:
        krw_ar      - Numpy array, contatining arithmetic averaged relative permeabilities 
                      at the face centers
    """
    
    z_cntr = g.cell_centers[2,:]     # z-values of cell centers
    z_fcs = g.face_centers[2,:]      # z-values of face centers 
    neu_fcs = bc.is_neu.nonzero()    # neumann boundary faces
    dir_fcs = bc.is_dir.nonzero()    # dirichlet boundary faces
    int_fcs = g.get_internal_faces() # internal faces

    fcs_neigh = np.zeros((g.num_faces,2),dtype=int) #          
    fcs_neigh[:,0] = g.cell_face_as_dense()[0]      # faces neighbouring mapping
    fcs_neigh[:,1] = g.cell_face_as_dense()[1]      # 
    
    int_fcs_neigh = fcs_neigh[int_fcs]              # internal faces neighbouring mapping
    
    # Initializing 
    krw_ar = np.zeros(g.num_faces)
    
    # Neumann boundaries relative permeabilities
    krw_ar[neu_fcs] = 1.
    
    # Dirichlet boundaries relative permeabilities
    dir_cells_neigh = fcs_neigh[dir_fcs] # neighboring cells of dirichlet faces
    dir_cells = dir_cells_neigh[(dir_cells_neigh >= 0).nonzero()] # cells that share a dirichlet face 
    krw_ar[dir_fcs] = 0.5 * ( \
                             krw(bc_val[dir_fcs]-z_fcs[dir_fcs]) + \
                             krw(h_m0[dir_cells]-z_cntr[dir_cells]) \
                            )
    
    # Internal faces relative permeabilities
    krw_ar[int_fcs] = 0.5 * ( \
                             krw(h_m0[int_fcs_neigh[:,0]] - z_cntr[int_fcs_neigh[:,0]]) + \
                             krw(h_m0[int_fcs_neigh[:,1]] - z_cntr[int_fcs_neigh[:,1]]) \
                            )
    
    return krw_ar

#%% Newton solver
            
def newton_solver(eq,h_ad,h_m,newton_param,time_param,print_param):
     R = eq.val                                                  # determining residual
     J = eq.jac                                                  # determining Jacobian
     y = sps.linalg.spsolve(J,-R)                                # Newton update
     h_ad.val +=  y                                              # root for the k-th step
     newton_param['abs_tol'] = np.max(np.abs(h_ad.val - h_m))    # checking convergence        
     
     if not print_param['is_active']:
         
         if newton_param['abs_tol'] <= newton_param['max_tol'] and newton_param['iter'] <= newton_param['max_iter']:
             print('Time: {0:4.2f} [s] \t Iter: {1:1d} \t Error: {2:4.3f} [cm]'.format(time_param['time_cum'],newton_param['iter'],newton_param['abs_tol']))
         elif newton_param['iter'] > newton_param['max_iter']:
             print('Error: Newton method did not converge!')
         else:
             newton_param['iter'] += 1
    
     else:    
         
        if newton_param['abs_tol'] <= newton_param['max_tol'] and newton_param['iter'] <= newton_param['max_iter']:
             print('\t\t Saving solutions: \t\t')
             print('Time: {0:4.2f} [s] \t Iter: {1:1d} \t Error: {2:4.3f} [cm]'.format(print_param['times'][print_param['counter']-1],newton_param['iter'],newton_param['abs_tol']))
        elif newton_param['iter'] > newton_param['max_iter']:
             print('Error: Newton method did not converge!')
        else:
             newton_param['iter'] += 1

#%% Saving routine
         
def save_solution(sol,newton_param,time_param,print_param,g,h_ad,h_m,theta,q):
    
    # Determining cell centers of the z-axis
    z_cntr = g.cell_centers[2,:]
    
    # Saving solutions corresponding to the initial conditions
    if time_param['time_cum'] == 0:
        sol['time'][print_param['counter']] = time_param['time_cum']
        sol['hydraulic_head'][print_param['counter']] = h_ad.val
        sol['pressure_head'][print_param['counter']] = h_ad.val - z_cntr
        sol['elevation_head'][print_param['counter']] = z_cntr
        sol['water_content'][print_param['counter']] = theta(h_ad.val-z_cntr)
        sol['darcy_fluxes'][print_param['counter']] = q(h_ad.val,h_ad.val)
           
    # Saving iterations and time step information at each level
    sol['iterations'] = np.concatenate((sol['iterations'],np.array([newton_param['iter']])))
    sol['time_step'] = np.concatenate((sol['time_step'],np.array([time_param['dt']])))    
       
    # Saving solutions at the printing levels
    if time_param['dt'] + time_param['time_cum'] >= print_param['times'][print_param['counter']]:
        sol['time'][print_param['counter']+1] = print_param['times'][print_param['counter']]
        sol['hydraulic_head'][print_param['counter']+1] = h_ad.val
        sol['pressure_head'][print_param['counter']+1] = h_ad.val - z_cntr
        sol['elevation_head'][print_param['counter']+1] = z_cntr
        sol['water_content'][print_param['counter']+1] = theta(h_ad.val-z_cntr)
        sol['darcy_fluxes'][print_param['counter']+1] = q(h_ad.val,h_m)
        print_param['counter'] += 1
        
def time_stepping(time_param,newton_param,print_param):
    
    """
    Time stepping algorithm... This need some work! 
    We need to make the time step step adaptation independent 
    from the printing routine
    
    Update: I could make the printing routine independent of the time stepping
            routine. Maybe still need some work with structure and algorithm
            efficiency
    """
    dt_aux = time_param['dt'] # this is an auxiliary variable
     
    # Increasing time step if we have a low number of iterations
    if newton_param['iter'] <= time_param['lower_opt_iter_range']:
        
        dt_aux *= time_param['lower_mult_factor']
        
        if time_param['time_cum'] + dt_aux > time_param['sim_time']:
            time_param['dt'] = time_param['sim_time'] - time_param['time_cum']
        elif dt_aux > time_param['dt_max']: 
            time_param['dt'] = time_param['dt_max']
        else:
            time_param['dt'] = dt_aux
    
    # Decreasing time if we have a high number of iterations
    elif newton_param['iter'] >= time_param['upper_opt_iter_range']:
        
        dt_aux *= time_param['upper_mult_factor']
        
        if time_param['time_cum'] + dt_aux > time_param['sim_time']:
            time_param['dt'] = time_param['sim_time'] - time_param['time_cum']
        elif dt_aux < time_param['dt_min']:
            time_param['dt'] = time_param['dt_min']
        else:
            time_param['dt'] = dt_aux
    
    # Time step remains unchanged
    else:
        
        if time_param['time_cum'] + dt_aux > time_param['sim_time']:
            time_param['dt'] = time_param['sim_time'] - time_param['time_cum']
        else:
            time_param['dt'] = dt_aux    
    
    # For the final step, adapt time step so it matches with final time
    if time_param['time_cum'] + time_param['dt'] > time_param['sim_time']:
        
        time_param['dt'] = time_param['sim_time'] - time_param['time_cum']
    
    