import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

def count_D_alpha(arrival_times, alpha):
    
    counter = 0
    for t in arrival_times:
        if t>alpha:
            break
        
        counter +=1
        
    return counter


ps = [1/2, 1/21, 1/61]

alphas = [1/3, 2/3]

rng = np.random.default_rng(seed = 0)

n_samples = 100000

for p in ps:
    for alpha in alphas:
        q = 1 - (1-p)*(1-alpha)
        D = []
        D_alpha = []
        D_remaining = []
        for _ in range(n_samples):
        
            realization = rng.geometric(p) -1
            D.append(realization)
            
            arrival_times =  np.sort( rng.uniform(size = realization) )        
            
            counter = count_D_alpha(arrival_times, alpha)
            D_alpha.append(counter)
            D_remaining.append(realization-counter)
        
        D = np.array(D)
        D_alpha =np.array(D_alpha)
        D_remaining = np.array(D_remaining)
        
        reg = LinearRegression().fit(D_alpha.reshape(-1, 1), D_remaining)
        print(alpha, p)
        print(f'Regression formula: E[D_rem|D_alpha] = {reg.coef_[0]:.3f}*D_alpha + {reg.intercept_:.3f}')
        print(f'Analytical formula: E[D_rem|D_alpha] = {(1-q)/q:.3f}*D_alpha + {(1-q)/q:.3f}')
    # plt.scatter(D, D13)
    # plt.ylabel('D13')
    # plt.xlabel('D')
    # plt.show()

