# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rd

def Mackey_Glass_equation(init,length,K,beta=0.25,gamma=0.1,tau=17,n=10):
    u = [0] * tau + [init]
    for i in range(length-1):
        du = beta * u[-1-tau] / (1 + u[-1-tau] ** n) - gamma * u[-1]
        next_u = u[-1] + du
        u.append(next_u)
    del u[:tau]
    u = np.array(u,dtype=np.float32).reshape(length,K,1)
    return(u)
