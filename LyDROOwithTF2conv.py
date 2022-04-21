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

# for tensorflow2
from memoryTF2conv import MemoryDNN
# from optimization import bisection
from ResourceAllocation import Algo1_NUM
from system_params import *
from User import User

import time

import os 

def create_img_folder(): 
    path = f'./V=1e{V},dth={d_th},lambda={lambda_param},n={n}/'
    os.makedirs(path, exist_ok=True)
    print(f"Directory {os.getcwd()}")
    return path 

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
    # plt.plot(np.arange(len(rate_array))+1, np.hstack(df.rolling(rolling_intv, min_periods=1).mean().values), 'b')
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
    V = 10**V

    K = N                 # initialize K = N

    arrival_lambda = lambda_param*np.ones((N)) # 1.5 Mbps per user

    print('#user = %d, #channel=%d, K=%d, decoder = %s, Memory = %d, Delta = %d'%(N,n,K,decoder_mode, Memory, Delta))

    # initialize data
    channel = np.zeros((n,N)) # chanel gains
    dataA = np.zeros((n,N)) # arrival data size

    init_location = [(110, np.pi/4, np.pi, 1.5), 
        (110, np.pi*3/4, 0, 0),
        (10, 0, np.pi*3/4, 0.9),
        (80, -3/4*np.pi, np.pi/6, 1.5),
        (110, -3/4*np.pi, np.pi/12, 1.5),
        (110, np.pi/4, np.pi, 1.5), 
        (110, np.pi*3/4, 0, 0),
        (10, 0, np.pi*3/4, 0.9),
        (80, -3/4*np.pi, np.pi/6, 1.5),
        (110, -3/4*np.pi, np.pi/12, 1.5)]

    users = [User(iloc) for iloc in init_location]
    
    
    mem = MemoryDNN(net = [N*3, 256, 128, N],
                    learning_rate = 0.01,
                    training_interval=10,
                    batch_size=128,
                    memory_size=Memory)

    start_time=time.time()
    mode_his = [] # store the offloading mode
    k_idx_his = [] # store the index of optimal offloading actor
    Q = np.zeros((n,N)) # local queue in tasks
    L = np.zeros((n,N)) # UAV queue in tasks
    D = np.zeros((n,N)) # delay in time slots

    Y = np.zeros((n,N)) # virtual energy queue in mJ
    Obj = np.zeros((n)) # objective values after solving problem (26)
    energy = np.zeros((n,N)) # energy consumption
    rate = np.zeros((n,N)) # achieved computation rate
    d_t = np.zeros((n, N)) 
    
    a = np.zeros((n, N)) # number of local computation tasks 
    b = np.zeros((n, N)) # number of offloading tasks 
    c = np.zeros((n, N))  # number of remote computation tasks
    delay = np.zeros((n, N)) # estimated delay 



    for i in range(n):

        if i % (n//10) == 0:
            print("%0.1f"%(i/n))
        if i> 0 and i % Delta == 0:
            # index counts from 0
            if Delta > 1:
                max_k = max(np.array(k_idx_his[-Delta:-1])%K) +1
            else:
                max_k = k_idx_his[-1] +1
            K = min(max_k +1, N)

        i_idx = i



        # #real-time channel generation
        # h_tmp = racian_mec(h0,0.3)
        # # increase h to close to 1 for better training; it is a trick widely adopted in deep learning
        # h = h_tmp*CHFACT
        # h = channel_model()
        h_tmp = [dB(iuser.channel_gain_dB[i]) for iuser in users] 
        h = h_tmp
            
        channel[i,:] = h
        # real-time arrival generation
        dataA[i,:] = np.random.poisson(arrival_lambda, size=(1, N))


        # 4) ‘Queueing module’ of LyDROO
        if i_idx > 0:
            # update queues
            Q[i_idx,:] = np.maximum(0, Q[i_idx-1,:] - a[i_idx-1,:] - b[i_idx-1,:]) + dataA[i_idx-1,:] # current data queue
            L[i_idx,:] = np.maximum(0, L[i_idx-1,:] - c[i_idx-1,:]) + b[i_idx-1,:]
            D[i_idx,:] = D[i_idx-1,:] + scale_delay*delay[i_idx-1, :] - scale_delay*d_th
            

        # scale Q and Y to 1
        nn_input =np.vstack( (h, Q[i_idx,:],L[i_idx,:])).transpose().flatten()


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

            # estimate the current value delay
            Q_i_t = np.zeros(N)
            L_i_t = np.zeros(N)
            b_i_t = np.zeros(N)
            # avarage local queue 
            if i_idx > 0: 
                begin_idx = i_idx - 10 if i_idx > 10 else 0
                Q_i_t = np.mean(Q[begin_idx:i_idx+1, :], axis=0) 
                # average uav queue 
                L_i_t = np.mean(L[begin_idx:i_idx+1, :], axis=0)
                # average arrival rate at remote queue 
                b_i_t = np.mean(b[begin_idx:i_idx, :], axis=0)

            d_i_t = Q_i_t/arrival_lambda + (1 - m)*1
        
            for iuser, bt in enumerate(b_i_t): 
                if bt > 0: 
                    d_i_t[iuser] = d_i_t[iuser] + m[iuser]*(1 + L_i_t[iuser]/bt)

            # update the objective function
            # f_val = f_val + scale_delay*np.sum(1/2 * d_i_t**2 + d_i_t*(D[i_idx,:] - d_th))

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
        tmp, a[i_idx,:],b[i_idx,:],c[i_idx,:], energy[i_idx, :] = r_list[k_idx_his[-1]]  

        print(f'local computation: a_i =', a[i_idx,:])
        print(f'offloading volume: b_i =', b[i_idx,:])
        print(f'remote computation: c_i =', c[i_idx,:])
        print(f'remote computation: energy_i =', energy[i_idx,:])
        print(f'fvalue = {v_list[k_idx_his[-1]]}')


    total_time=time.time()-start_time
    print(f'total time: {total_time}')
    mem.plot_cost(path_name=path+'TraningLoss')

    plot_rate(Q.sum(axis=1)/N, 100, 'User queue length', name=path+'UserQueue')
    plot_rate(L.sum(axis=1)/N, 100, 'UAV queue length', name=path+'UAVQueue')
    plot_rate(energy.sum(axis=1)/N/delta*1000, 100, 'Power consumption (mW)', name=path+'AvgPower')
    plot_rate(delay.sum(axis=1)/N, 100, 'Latency (TS)',name=path+'AvgDelay')

    print('Average time per channel:%s'%(total_time/n))

    # save all data
    sio.savemat(path+'./result_%d.mat'%N, {'input_h': channel/CHFACT,'data_arrival':dataA,'local_queue':Q,'uav_queue':L,'off_mode':mode_his,'energy_consumption':energy,'delay':delay,'objective':Obj})
    print('completed!')