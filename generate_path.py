import sys
import numpy as np
import matplotlib.pyplot as plt

gamma = 20
D = 1#intense of noise
K = 1#intense of potential
dt = 10**-3#time step 
dx = 1e-2#微分する用のdx
steps = int(10**3)#ステップ数
beta = 2#逆温度
tau = steps * dt#time duration of a path


#fにはx,tの2引数を持つ関数を渡す
def simulate_Langevin(f, back=True, x0=0):
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
        return forward_path, 0
    
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
def generate_datasets(f, n, label, back=True):
    datasets = []
    labels = []
    for i in range(n):
        forward, backward = simulate_Langevin(f, back = back)
        datasets.append(np.append(forward, 0))
        if back:
            datasets.append(np.append(backward, 1))
        if (i+1)*10 % n== 0:
            print("{}/{} generated".format(i+1, n))
    return datasets, [label for i in range(len(datasets))]

def which_f(f1, f2, trajectory):
    #trajectoryが
    #f1から生成された確率
    tr = trajectory
    n = len(trajectory)
    logp0 = -1/(2*dt)*sum([(tr[i+1] -tr[i])**2 for i in range(n-1)])
    logL1 = -0.5*(sum([f1(tr[i], i*dt)**2 * dt for i in range(n)]) + sum([f1(tr[i], i*dt)*(tr[i+1]-tr[i])/D for i in range(n-1)]))
    logL2 = -0.5*(sum([f2(tr[i], i*dt)**2 * dt for i in range(n)]) + sum([f2(tr[i], i*dt)*(tr[i+1]-tr[i])/D for i in range(n-1)]))
    logpf1, logpf2 = logp0+logL1, logp0+logL2
    #print(logpf1, logpf2)
    #p1, p2 = np.exp(logpf1), np.exp(logpf2)
    return np.exp(logL1)/(np.exp(logL1)+np.exp(logL2))

if __name__ == '__main__':
    f1 = lambda x, t:2*x
    f2 = lambda x, t:3*x**3
    args = sys.argv
    n_train = int(args[1])
    n_test  = int(args[2])
    back = True
    if len(args)>3:
        back = False if args[3]=="False" else True 
    #実際にはその4倍返ってくる
    #順f1, 逆f1, 順f2, 逆f2
    
    traindata_X1, traindata_Y1 = generate_datasets(f1, n_train, 0, back = back)
    traindata_X2, traindata_Y2 = generate_datasets(f2, n_train, 1, back = back)
    testdata_X1, testdata_Y1 = generate_datasets(f1, n_test, 0, back = back)
    testdata_X2, testdata_Y2 = generate_datasets(f2, n_test, 1, back = back)
    
    
    
    traindata_X = np.array(traindata_X1+traindata_X2)
    traindata_Y = np.array(traindata_Y1+traindata_Y2)
    testdata_X = np.array(testdata_X1+testdata_X2)
    testdata_Y = np.array(testdata_Y1+testdata_Y2)
    
    print("{} trajectories were generated".format(len(traindata_X)+len(testdata_X)))
    correct = 0
    for i in range(len(testdata_X)):
        f1likelihood = which_f(f1, f2, testdata_X[i])
        label = testdata_Y[i]
        if f1likelihood > 0.5 and label == 0:
            correct += 1
        elif f1likelihood <0.5 and label == 1:
            correct += 1
    print("theoretical correct rate =", correct/len(testdata_X))



    np.savez("traindata_X.npz", traindata_X)
    np.savez("traindata_Y.npz", traindata_Y)
    np.savez("testdata_X.npz", testdata_X)
    np.savez("testdata_Y.npz", testdata_Y)
    
    #tmp = (np.load("traindata_Y.npz"))
    #for i in tmp.keys():
    #    print(tmp['arr_0'])
    
    
    
