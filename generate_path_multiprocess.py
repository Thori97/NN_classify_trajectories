import sys
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing as multi

gamma = 20
D = 1#intense of noise
K = 1#intense of potential
dt = 10**-3#time step 
dx = 1e-2#微分する用のdx
steps = int(10**3)#ステップ数
beta = 2#逆温度
tau = steps * dt#time duration of a path


#fにはx,tの2引数を持つ関数を渡す
def simulate_Langevin(f, x0=0, back=True):
    #順軌跡
    forward_path = np.zeros(steps)
    t = 0
    x = x0
    forward_path[0] = x
    for i in range(steps-1):
        dxdt = -1*f(x, t) + (2*D/dt)**0.5*np.random.normal(0, 1)#ノイズをdt積分するとdt**0.5かかる, あとでdtかけられるからここでさらに1/dtする
        x += dxdt*dt
        forward_path[i+1] = x
        t += dt
    
    if not back:
        return forward_path
    
    #逆軌跡
    backward_path = np.zeros(steps)
    t = tau
    #x = x0
    backward_path[0] = x
    for i in range(steps-1):
        dxdt = -1*f(x, t) + (2*D/dt)**0.5*np.random.normal(0, 1)#ノイズをdt積分するとdt**0.5かかる, あとでdtかけられるからここでさらに1/dtする
        x += dxdt*dt
        backward_path[i+1] = x
        t -= dt
    return forward_path, backward_path[::-1]

#返り値 [data1, data2, ..., datan], [label1, label2, ..., labeln]
def generate_datasets(f, n, label):
    datasets = []
    labels = []
    for i in range(n):
        forward, backward = simulate_Langevin(f)
        datasets.append(np.append(forward, 0))
        datasets.append(np.append(backward, 1))
    return datasets, [label for i in range(2*n)]

if __name__ == '__main__':
    f1 = lambda x, t:4*x
    f2 = lambda x, t:4*x**3
    args = sys.argv
    n_train = int(args[1])
    n_test  = int(args[2])
    #実際にはその4倍返ってくる
    #順f1, 逆f1, 順f2, 逆f2
    
    traindata_X1, traindata_Y1 = generate_datasets(f1, n_train, 0)
    traindata_X2, traindata_Y2 = generate_datasets(f2, n_train, 1)
    testdata_X1, testdata_Y1 = generate_datasets(f1, n_test, 0)
    testdata_X2, testdata_Y2 = generate_datasets(f2, n_test, 1)
    
    print("{} trajectories were generated".format(n_train*4+n_test*4))
    
    
    traindata_X = np.array(traindata_X1+traindata_X2)
    traindata_Y = np.array(traindata_Y1+traindata_Y2)
    testdata_X = np.array(testdata_X1+testdata_X2)
    testdata_Y = np.array(testdata_Y1+testdata_Y2)
    
    np.savez("traindata_X.npz", traindata_X)
    np.savez("traindata_Y.npz", traindata_Y)
    np.savez("testdata_X.npz", testdata_X)
    np.savez("testdata_Y.npz", testdata_Y)
    
    #tmp = (np.load("traindata_Y.npz"))
    #for i in tmp.keys():
    #    print(tmp['arr_0'])
    
    
    
