import gurobipy as gp
from gurobipy import GRB
import networkx

def LP(m,p,q,w):
    
    mod = gp.Model('star')
    
    x = {}
    s = {}
    for theta in range(m):
        
        s[theta] = mod.addVar(name = f's_{theta}')
        
        for j in range(m):
            x[j,theta] = mod.addVar(name = f'x_{j}_{theta}', lb = 0)
     


            
    for theta in range(m):
    
        
        for j in range(m):
            mod.addConstr( gp.quicksum( x[j,theta_prime] for theta_prime in range(theta,m) ) <= s[theta] )
            
        mod.addConstr(gp.quicksum(x[j,theta] for j in range(m)) <= s[theta])
        
        if theta == 0 :
            mod.addConstr( s[theta] == 1 )
        else:
            mod.addConstr( s[theta] == (s[theta-1] - gp.quicksum(p[j]*x[j,theta-1] for j in range(m)) ) * q[theta]/q[theta-1] )
    
    mod.setObjective( gp.quicksum( w[j]*p[j]*x[j,theta] for j in range(m) for theta in range(m)) , GRB.MAXIMIZE)
    
    
    return mod,x,s
    
if __name__ == '__main__':
    print('sup')
    
    q = [1,1/3]
    m = 2
    p = [3/4,1/4]
    
    w = [1,2]
    
    
    epsilon = 0.0001
    
    w = [1,0]
    
    p = [epsilon,1]
    
    q = [1,1]
    
    
    
    mod,x,s = LP(m,p,q,w)
    
    mod.optimize()