import numpy as np
import random
import math

def Load_data(filedir):
    """
    Function: Download the data
    """
    user_set = {}
    item_set = {}
    N = 0;    #the number of user
    M = 0;    #the number of item
    user_id = 0
    item_id = 0
    data = []
    f = open(filedir)
    for line in f.readlines():
        u,i,score,time = line.split()
        if int(u) not in user_set:
            user_set[int(u)] = user_id
            user_id = user_id + 1
        if int(i) not in item_set:
            item_set[int(i)] = item_id
            item_id = item_id + 1
        data.append([user_set[int(u)],item_set[int(i)],int(score)])
    f.close()
    N = user_id;
    M = item_id;

    np.random.shuffle(data)
    #train = data[0:int(len(data)*ratio)]
    #test = data[int(len(data)*ratio):]
    return N,M,data


def SGD(train,test,N,M,eta,K,lambda_1,lambda_2,SGD_length):
    """
    train: train data
    test: test data
    N:the number of user
    M:the number of item
    eta: the learning rata
    K: the number of latent factor
    lambda_1,lambda_2: regularization parameters of user or item
    Step: the max iteration
    """
    U = np.random.normal(0, 0.1, (N, K))    #the latent factor of user
    V = np.random.normal(0, 0.1, (M, K))    #the latent factor of item
    L = 1000.0                              #the end of SGD
    rmse = []                               #RMSE
    loss = []
    for ste in range(SGD_length):
        los = 0.0
        for data in train:
            u = data[0]
            i = data[1]
            score = data[2]

            e = score - np.dot(U[u],V[i].T)            
            U[u] = U[u] + eta*(e*V[i] - lambda_1*U[u])
            V[i] = V[i] + eta*(e*U[u] - lambda_2*V[i])

            los = los + 0.5*(e**2 + lambda_1*np.square(U[u]).sum() + lambda_2*np.square(V[i]).sum())     #the loss function
        loss.append(los)
        rms = RMSE(U,V,test)
        rmse.append(rms)
        if los<L:
            break

    return loss,rmse,U,V

           
def RMSE(U,V,test):
    """
    Founction: Calculate the RMSE
    U: the prediced latent factor of user
    V: the prediced latent factor of user
    test:  test data
    """
    count = len(test)
    sum_rmse = 0.0
    for t in test:
        u = t[0]
        i = t[1]
        r = t[2]
        pr = np.dot(U[u],V[i].T)
        sum_rmse += np.square(r-pr)
    rmse = np.sqrt(sum_rmse/count)
    return rmse



def Figure(loss,rmse):
    fig1=plt.figure('LOSS')
    x = range(len(loss))
    plot(x, loss, color='g',linewidth=3)
    plt.title('PMF')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    fig2=plt.figure('RMSE')
    x = range(len(rmse))
    plot(x, rmse, color='r',linewidth=3)
    plt.title('PMF')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    show()

 
def main():
    dir_data = "./u.data"
    dir_train_data = './ua.base'
    dir_test_data = './ua.test'
    #ratio = 0.8
    N,M,data = Load_data(dir_data)   # all data
    print('用户的个数：',N)
    print('项目的个数：',M)
 
    train_N,train_M,train_data = Load_data(dir_train_data)
    test_N,test_M,test_data = Load_data(dir_test_data)
        
    eta = 0.005
    K = 10
    lambda_1 = 0.1
    lambda_2 = 0.1
    SGD_length = 50
    loss,rmse,U,V = SGD(train_data,test_data,N,M,eta,K,lambda_1,lambda_2,SGD_length)
    print ('PMF误差精度为：',rmse[-1])   #caculate the error
    Figure(loss,rmse)
    
    
         
if __name__ == '__main__': 
    main()
