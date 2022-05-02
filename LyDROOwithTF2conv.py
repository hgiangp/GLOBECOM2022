#  #################################################################
#
#  This file contains the main code of LyDROO.
#
#  References:
#  [1] Suzhi Bi, Liang Huang, Hui Wang, and Ying-Jun Angela Zhang, "Lyapunov-guided Deep Reinforcement Learning for Stable Online Computation Offloading in Mobile-Edge Computing Networks," IEEE Transactions on Wireless Communications, 2021, doi:10.1109/TWC.2021.3085319.
#  [2] Liang Huang, Suzhi Bi, and Ying-Jun Angela Zhang, "Deep Reinforcement Learning for Online Offloading in Wireless Powered Mobile-Edge Computing Networks," in IEEE Transactions on Mobile Computing, vol. 19, no. 11, pp. 2581-2593, November 2020.
#  [3] S. Bi and Y. J. Zhang, “Computation rate maximization for wireless powered mobile-edge computing with binary computation ofﬂoading,” IEEE Trans. Wireless Commun., vol. 17, no. 6, pp. 4177-4190, Jun. 2018.
#
# version 1.0 -- July 2020. Written by Liang Huang (lianghuang AT zjut.edu.cn)
#  #################################################################


import scipy.io as sio                     # import scipy.io for .mat file I/
import numpy as np                         # import numpy
import pandas as pd 

# for tensorflow2
from memoryTF2conv import MemoryDNN
# from optimization import bisection
from ResourceAllocation import Algo1_NUM
from system_params import d_th, scale_delay, V
from User import *

import time

import os 

def create_img_folder(): 
    # path = f'./V={V},dth={d_th},lambda={lambda_param},T={T},W={W},scale_delay={scale_delay}, psi = {psi}/'
    path = "{}/img_new/V ={:.2e},dth={:},lambda={:},T={:},W={:.2e},scale_delay={:}, psi = {:}, f_u ={:.2e}/".format(os.getcwd(),
     V, d_th, lambda_param, T, W, scale_delay, psi, f_u_max)
    os.makedirs(path, exist_ok=True)
    print(f"Directory {os.getcwd()}")
    return path

def plot_drift(Q, L, D, E, name, rolling_intv=50): 
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    # mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(8,8))
    plt.grid()
    plt.xlim([1, len(Q)+1])

    rate_array = np.asarray(Q)
    df = pd.DataFrame(Q)
    # plt.plot(np.arange(T), Q, np.arange(T), L, np.arange(T), D, np.arange(T), E)
    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values))

    df = pd.DataFrame(L)
    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values))
    df = pd.DataFrame(D)
    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values))
    df = pd.DataFrame(E)
    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values))



    plt.xlabel('Time Frames')
    plt.legend(['Q','L', 'D', 'E'])
    plt.savefig(name)
    # plt.show()


def plot_rate( rate_his, rolling_intv = 50, ylabel='Normalized Computation Rate', name='Average queue length'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import matplotlib as mpl

    rate_array = np.asarray(rate_his)
    df = pd.DataFrame(rate_his)


    # mpl.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(15,8))
    plt.grid()
    plt.plot(np.arange(len(rate_array))+1, rate_array)
    plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), 'b')
    # plt.fill_between(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).min()[0].values), np.hstack(df.rolling(rolling_intv, min_periods=1).max()[0].values), color = 'b', alpha = 0.2)
    plt.ylabel(ylabel)
    plt.xlabel('Time Frames')
    plt.savefig(name)
    plt.show()

if __name__ == "__main__":
    '''
        LyDROO algorithm composed of four steps:
            1) 'Actor module'
            2) 'Critic module'
            3) 'Policy update module'
            4) ‘Queueing module’ of
    '''

    path = create_img_folder()
    # V = 10**V

    K = N                   # initialize K = N

    arrival_lambda = lambda_param*np.ones((N)) # 1.5 Mbps per user

    print('#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d'%(N,T,K,decoder_mode, Memory, Delta))

    # initialize data
    channel = np.zeros((T,N)) # chanel gains
    dataA = np.zeros((T,N)) # arrival data size

    init_location = [[110, np.pi/4, np.pi, 1.5], 
        [110, np.pi*3/4, 0, 0],
        [10, 0, np.pi*3/4, 0.9],
        [80, -3/4*np.pi, np.pi/6, 1.5],
        [110, -3/4*np.pi, np.pi/12, 1.5],
        [110, np.pi/4, np.pi, 1.5], 
        [110, np.pi*3/4, 0, 0],
        [10, 0, np.pi*3/4, 0.9],
        [80, -3/4*np.pi, np.pi/6, 1.5],
        [110, -3/4*np.pi, np.pi/12, 1.5]]

    users = [User(iloc) for iloc in init_location]
    
    # for iuser in users: 
    #     iuser.generate_channel_gain()
    
    mem = MemoryDNN(net = [N*4, 256, 128, N],
                    learning_rate = 0.01,
                    training_interval=20,
                    batch_size=128,
                    memory_size=Memory)

    start_time=time.time()
    mode_his = [] # store the offloading mode
    k_idx_his = [] # store the index of optimal offloading actor
    Q = np.zeros((T,N)) # local queue in tasks
    L = np.zeros((T,N)) # UAV queue in tasks
    D = np.zeros((T,N)) # delay in time slots

    Obj = np.zeros((T)) # objective values after solving problem (26)
    energy = np.zeros((T,N)) # energy consumption
    energy_uav = np.zeros((T))
    rate = np.zeros((T,N)) # achieved computation rate
    d_t = np.zeros((T, N)) 
    
    a = np.zeros((T, N)) # number of local computation tasks 
    b = np.zeros((T, N)) # number of offloading tasks 
    c = np.zeros((T, N))  # number of remote computation tasks
    delay = np.zeros((T, N)) # estimated delay 

    drift_Q = np.zeros((T))
    drift_L = np.zeros((T))
    drift_D = np.zeros((T))
    weighted_energy = np.zeros((T))



    for i in range(T):

        if i % (T//10) == 0:
            print("%0.1f"%(i/T))
        if i> 0 and i % Delta == 0:
            # index counts from 0
            if Delta > 1:
                max_k = max(np.array(k_idx_his[-Delta:-1])%K) +1
            else:
                max_k = k_idx_his[-1] +1
            K = min(max_k +1, N)
        # K = 40

        i_idx = i



        # #real-time channel generation
        # h_tmp = racian_mec(h0,0.3)
        # # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
        # h = h_tmp*CHFACT
        # h = channel_model()
        h_tmp = np.array([dB(iuser.gain_dB[i]) for iuser in users])
        h = h_tmp
            
        channel[i,:] = h
        # real-time arrival generation
        # dataA[i,:] = np.random.poisson(arrival_lambda, size=(1, N))
        dataA[i,:] = np.round(np.random.uniform(low=0, high=arrival_lambda*2, size=(1, N)))


        # 4) ‘Queueing module’ of LyDROO
        if i_idx > 0:
            # update queues
            Q[i_idx,:] = np.maximum(0, Q[i_idx-1,:] - a[i_idx-1,:] - b[i_idx-1,:]) + dataA[i_idx-1,:] # current data queue
            L[i_idx,:] = np.maximum(0, L[i_idx-1,:] - c[i_idx-1,:]) + b[i_idx-1,:]
            D[i_idx,:] = np.maximum(0, D[i_idx-1,:] + scale_delay*delay[i_idx-1, :] - scale_delay*d_th)

            drift_Q[i_idx] = 1/2 *np.sum((Q[i_idx,:] - Q[i_idx - 1,:])**2)
            drift_L[i_idx] = 1/2 *np.sum((L[i_idx,:] - L[i_idx - 1,:])**2)
            drift_D[i_idx] = 1/2 *np.sum((D[i_idx,:] - D[i_idx - 1,:])**2)

            

        # scale Q and Y to 1
        h_norm = np.linalg.norm(h)
        
        q_norm = np.linalg.norm(Q[i_idx,:])
        l_norm = np.linalg.norm(L[i_idx,:])
        d_norm = np.linalg.norm(D[i_idx,:])

        if q_norm == 0: 
            q_norm = 1 
        if l_norm == 0: 
            l_norm = 1 
        if d_norm == 0: 
            d_norm = 1 
        
        # q_norm = 500
        # l_norm = 500
        # d_norm = d_th*scale_delay
        # h_norm = 1e-12

        nn_input =np.vstack((h/h_norm, Q[i_idx,:]/q_norm, L[i_idx,:]/l_norm,D[i_idx, :]/d_norm)).transpose().flatten()


        # 1) 'Actor module' of LyDROO
        # generate a batch of actions
        m_list = mem.decode(nn_input, K, decoder_mode)
 
        r_list = [] # all results of candidate offloading modes
        v_list = [] # the objective values of candidate offloading modes
        delay_list = []
        for m in m_list:
            # 2) 'Critic module' of LyDROO
            # allocate resource for all generated offloading modes saved in m_list
            r_list.append(Algo1_NUM(m,h,Q[i_idx,:],L[i_idx,:],V))
            # v_list.append(r_list[-1][0])
            f_val = r_list[-1][0]

            tmp, a_i, b_i, c_i, energy_i, energy_uav_i = r_list[-1]

            # estimate the current value delay
            Q_i_t = np.zeros(N)
            L_i_t = np.zeros(N)
            b_i_t = np.zeros(N)
            # avarage local queue 

            # b_idx = np.maximum(0, i_idx - 10) 
            b_idx = 0 
            Q_i_t = np.mean(Q[b_idx:i_idx+1, :], axis=0) 
            # average uav queue 
            L_i_t = np.mean(L[b_idx:i_idx+1, :], axis=0)
            # average arrival rate at remote queue 
            b_i_t = np.mean(b[b_idx:i_idx+1, :], axis=0)

            d_i_t = Q_i_t/arrival_lambda + (1 - m)*1 + m*(L_i_t/50)
            # d_i_t = Q_i_t/arrival_lambda + (1 - m)*1 + m*(L_i_t/50 + 2)

        
            # for iuser, bt in enumerate(b_i_t): 
            #     if m[iuser] == 1 and bt != 0: 
            #         d_i_t[iuser] = d_i_t[iuser] + m[iuser] *(2 + L_i_t[iuser]/bt)

            # update the objective function
            f_val = f_val + np.sum(1/2*(scale_delay*d_i_t)**2 + scale_delay*d_i_t*(D[i_idx,:] - scale_delay*d_th))
            

            v_list.append(f_val)
            delay_list.append(d_i_t)
            
        # record the index of largest reward
        k_idx_his.append(np.argmin(v_list))

        # 3) 'Policy update module' of LyDROO
        # encode the mode with largest reward
        mem.encode(nn_input, m_list[k_idx_his[-1]])
        mode_his.append(m_list[k_idx_his[-1]])

        # store max result
        # Obj[i_idx],rate[i_idx,:],energy[i_idx,:]  = r_list[k_idx_his[-1]]
        Obj[i_idx] = v_list[k_idx_his[-1]]
        delay[i_idx] = delay_list[k_idx_his[-1]]
        tmp, a[i_idx,:],b[i_idx,:],c[i_idx,:], energy[i_idx, :], energy_uav[i_idx] = r_list[k_idx_his[-1]]

        

        # drifted energy 
        weighted_energy[i_idx] = (np.sum(energy[i_idx, :]) + energy_uav[i_idx])*V

        print(f'local computation: a_i =', a[i_idx,:])
        print(f'offloading volume: b_i =', b[i_idx,:])
        print(f'remote computation: c_i =', c[i_idx,:])
        print(f'remote computation: energy_i =', energy[i_idx,:])
        print(f'fvalue = {v_list[k_idx_his[-1]]}')
        print('\n')


    total_time=time.time()-start_time
    print(f'total time = {total_time}')
    mem.plot_cost(path_name=path+'TraningLoss')

    

    plot_rate(Q.sum(axis=1)/N, 100, 'User queue length', name=path+'UserQueue')
    plot_rate(L.sum(axis=1)/N, 100, 'UAV queue length', name=path+'UAVQueue')
    plot_rate(energy.sum(axis=1)/N/delta*1000, 100, 'Power consumption (mW)', name=path+'AvgPower')
    plot_rate(delay.sum(axis=1)/N, 100, 'Latency (TS)',name=path+'AvgDelay')
    plot_rate(energy_uav/N/delta*1000, 100, 'UAV power consumption (mW)', name=path+'AvgPowerUAV')

    plot_drift(drift_Q, drift_L, drift_D, weighted_energy, name=path+"drift")
    
    print(f"Mean drift local queue: {np.mean(drift_Q)}")
    print(f"Mean drift uav queue: {np.mean(drift_L)}")
    print(f"Mean drift delay: {np.mean(drift_D)}")
    print(f"Mean drift energy: {np.mean(weighted_energy)}")

    print('Average time per channel:%s'%(total_time/T))

    # save all data
    aQ = np.mean(Q, axis=1)
    aL = np.mean(L, axis=1)
    aE_i = np.mean(energy, axis=1)
    aE_u = energy_uav
    adelay = np.mean(delay, axis=1)
    
    sio.savemat('./result_%d.mat'%N, {'input_h': channel/CHFACT,'data_arrival':dataA,'local_queue':Q,'uav_queue':L,'off_mode':mode_his,'energy_consumption':energy,'delay':delay,'objective':Obj})
    df = pd.DataFrame({'local_queue':aQ,'uav_queue':aL,'energy_user':aE_i,'energy_uav':aE_u, 'delay':adelay})
    name= path + 'V1.csv'
    df.to_csv(name, index=False)
    # return quequeue_rsue, energy
    print('completed!')
