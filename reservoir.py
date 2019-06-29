# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
import numpy.random as rd
import pandas as pd
import pickle

class Random():
    def __init__(self, output_folder, K, N, C):
        self.folder = output_folder
        self.K = K
        self.N = N
        self.C = C
        self.W_in = rd.normal(0,1,[K,1])
        W = rd.normal(0,1,[N,N])
        original_W = W
        mask = np.random.choice([0,1],size=[N,N],p=[1-self.C,self.C]).astype(np.float32)
        W = np.multiply(W,mask)
        eigs_W, _ = la.eig(W)
        sr_W = np.max(abs(eigs_W))
        self.W = W / sr_W
        #np.savez('{}/weight.npz'.format(output_folder),W_in=self.W_in,W=self.W,original_W=original_W)

    def _update(self, time, u):

        self.x = np.tanh(np.dot(self.W_in, u) + np.dot(self.W, self.x))

        self.reservoir = self.reservoir.append(pd.Series([time,u,self.x],index=self.reservoir.columns),ignore_index=True)

    def run(self, u, length):
        self.reservoir = pd.DataFrame(columns=['time','u','x'])

        init_x = rd.rand(self.N,1).astype(np.float32)
        self.x = init_x

        self.reservoir = self.reservoir.append(pd.Series([-1,np.NaN,self.x],index=self.reservoir.columns),ignore_index=True)

        for time in range(length):
            self._update(time, u[time])

        #numpy arrayに書き換え
        x = np.array([self.reservoir.loc[i,'x'] for i in range(len(self.reservoir)) if i != 0],dtype=np.float32)

        return(x,self.reservoir)
