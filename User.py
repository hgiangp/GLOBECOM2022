import numpy as np
from numpy.random import default_rng
rng = default_rng()
import matplotlib.pyplot as plt

from system_params import *

class User: 
    def __init__(self, init_location):
        
        self.loc = []
        self.gain_dB = []
        self.channal_gain_wo_fading_dB = []
        
        self.Q = np.zeros((T))

        self.generate_channel_gain(init_location)



    def generate_channel_gain(self, location): 
        '''
        generate channel gain in dB 
        '''
        # generate the location 
        initial_loc = Location(location[0],location[1])
        mov_angle = location[2]
        velocity = location[3]

        self.loc.append(initial_loc)

        mu, sigma = velocity*delta, 1/3*velocity*delta
        dr = rng.normal(mu, sigma, (T)) # different location for each time slot 
        dvarphi_arr = []

        for i_t in range(1, T): 
            curr_pos = self.loc[-1]

            # update delta_varphi and varphi angle 
            dvarphi = rng.normal(mov_angle - curr_pos.varphi, np.pi/3)
            
            dvarphi_arr.append(dvarphi)

            next_position = curr_pos.update(dr[i_t], dvarphi)
            self.loc.append(next_position)

        # calculate theta angle 
        theta = np.asarray([np.arctan(H_uav/position.r) for position in self.loc])

        # calculate the line of sight probability 
        p_LOS = 1./(1 + a_LOS*np.exp(-b_LOS*(theta - a_LOS)))

        # generate the small scale fading 
        h_tidle = rng.normal(mu_gain, sigma_gain, size=(T)) # dB

        # calculate the channel gain 
        distance = [(H_uav**2 + (position.r**2))**(gamma/2) for position in self.loc]
        gain_wo_fading_dB = todB(p_LOS + xi*(1 - p_LOS)) + g0 - todB(distance)
        
        self.gain_dB = gain_wo_fading_dB + h_tidle
        self.channal_gain_wo_fading_dB = gain_wo_fading_dB
    
    def plot_location(self): 
        
        xt = np.zeros((T))
        yt = np.zeros((T))
        for it, loc in enumerate(self.loc): 
            xt[it], yt[it] = loc.get_decart_pos() 
        print('start plotting') 
        plt.plot(xt[0], yt[0], 'o')
        plt.plot(xt, yt, '-', linewidth=1)

    def plot_channel_gain(self): 
        x = [t for t in range (T)]
        plt.plot(x, self.gain_dB, '-', x, self.channal_gain_wo_fading_dB, '--', linewidth=0.8)

    def optimize_frequency(self, ts):  
        '''
        optimize the CPU frequency for local computation in time slot ts  
        '''
        Qt = self.Q[ts]
        f_hat = np.minimum(f_i_max, Qt*F/delta)
        f_i_L = np.minimum(np.sqrt(Qt/(3*kappa*V*F)), f_hat)
        
        energy = kappa * f_i_L**3 * delta
        at = np.round(f_i_L*delta/F)
        obj_val = V*energy - Qt*at 
        
        return at, energy, obj_val

    def update_queue(self, t, a_t, b_t, A_t):
        '''
        update the local queue in time slot t
        In: local volume at t a_t 
            offloaded volume at t b_t 
            arrival volume A_t
        ''' 
        self.Q[t+1] = np.minimum(0, self.Q[t] - a_t - b_t) + A_t

class Location: 
    def __init__(self, r, varphi):
        self.r = r
        self.varphi = varphi

    def update(self, dr, dphi):
        r = self.r  
        r2 = r**2
        dr2 = dr**2
        next_r = np.sqrt(r2 + dr2 + 2*r*dr*np.cos(dphi))
        next_angle = np.arccos((r2 + next_r**2 - dr2)/2/r/next_r) + self.varphi
        # next_angle = self.varphi + dphi  
        return Location(next_r, next_angle)

    def update_location(self, dr, dphi):
        r = self.r  
        coeff = [1, -2*r*np.cos(dphi), r**2 - dr**2]
        root = np.roots(coeff)
        next_r = root[np.where(root>0)][0]
        next_varphi = self.varphi + dphi
        return Location(next_r, next_varphi) 
    
    def get_decart_pos(self): 
        return (self.r*np.cos(self.varphi), self.r*np.sin(self.varphi))

if __name__ == '__main__':
    init_location = (110, np.pi/4, np.pi, 1.5)
    a = User(init_location=init_location)
    a.plot_location()
    # a.plot_channel_gain()
    plt.show()
    print('Simulation complete')




        
