import numpy as np 

from scipy.optimize import minimize, Bounds, LinearConstraint
from User import User
from system_params import * 



class Server: 
    def __init__(self, users_location):
        self.L = np.zeros((N, T))
        self.users = []
        self.gain = [] 
        self.init_user(users_location=users_location)
        
    
    def init_user(self, user_location):
        self.users = [User(user_location[iloc]) for iloc in user_location]
         
    def update_queue(self, ts_t, comp_c, offloaded_b):
        self.L[:, ts_t + 1] = np.maximum(0, self.L[:, ts_t] - comp_c) + offloaded_b
    
    def optimize_off_vol(self, mode, ts_t): 
        off_idx = np.where(mode == 1)[0]
        offloaded_L = self.L[:, ts_t][off_idx]
        offloaded_Q = np.vstack(self.users[idx].gain_dB[ts_t] for idx in off_idx)
        gain = np.vstack(self.users[idx].gain_dB[ts_t] for idx in off_idx)

        # calculate maximum number of offloaded task 
        b_hat = np.minimum(offloaded_Q, np.round(W*delta/R * np.log2(1 + p_i_max*gain/N0/W)))
        sub = np.minimum(1, offloaded_Q - offloaded_L)
        off_b = np.maximum(0, np.minimum(np.round(W*delta/R * np.log2(sub/(V*N0*R*np.log(2)))), b_hat))
        energy = N0*W*delta/gain *(2**(off_b*R/W/delta) - 1)
        obj_val = V*energy - off_b*(offloaded_Q - offloaded_L)

        return off_b, energy, obj_val

    def optimize_uav_freq(self, mode, ts_t): 
        off_idx = np.where(mode == 1)[0]
        offloaded_L = self.L[:, ts_t][off_idx]
        gain = np.vstack(self.users[idx].gain_dB[ts_t] for idx in off_idx)
        
        def obj_func(f_iU):
            return np.sum(-offloaded_L*f_iU*delta/F + V*psi*kappa*delta*(f_iU**3))
        
        def obj_der(f_iU): 
            return -offloaded_L*delta/F + V*psi*kappa*3*(f_iU**2)
        
        def obj_hess(f_iU): 
            return V*kappa*psi*delta*6*f_iU

        bounds = Bounds(0, offloaded_L*F/delta)
        A = np.ones_like(off_idx)
        linear_cons = LinearConstraint(A, 0, f_u_max)
        f_init = np.ones_like(off_idx)*1e8 
        res = minimize(obj_func, f_init, method='trust-constr', jac=obj_der, hess=obj_hess, tol=1e-7,
                    constraints=[linear_cons], bounds=bounds)

        # update uav frequency 
        f_iU = res.x
        
        energy = psi*kappa*delta*(f_iU**3)
        c_i = np.round(f_iU*delta/F)

        obj_val = np.around(res.fun, decimals=6)


        return c_i, energy, obj_val

    # def calculate_mt(self, iter): 
    #     if iter % (n//10) == 0:
    #         print("%0.1f"%(iter/n))
    #     if iter > 0 and iter % Delta == 0:
    #         # index counts from 0
    #         if Delta > 1:
    #             max_k = max(np.array(k_idx_his[-Delta:-1])%K) +1
    #         else:
    #             max_k = k_idx_his[-1] + 1
    #         K = min(max_k +1, N)

    def running(self): 
        for i in range(T): 



        
