# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
import numpy.random as rd
import pandas as pd
import pickle

class Reservoir():
    def __init__(self, output_folder, K, N, C, topology):
        self.folder = output_folder
        self.K = K
        self.N = N
        self.C = C
        self.W_in = rd.normal(0,1,[K,1])
        W = rd.normal(0,1,[N,N])
        original_W = W
        if topology == 'random':
            mask = np.random.choice([0,1],size=[N,N],p=[1-self.C,self.C]).astype(np.float32)
            W = np.multiply(W,mask)
        elif topology == 'ring':
            mask = np.eye(N)
            W = np.multiply(W,mask)
            W = np.roll(W,1,axis=1)
        elif topology == 'center':
            mask = np.zeros([N,N])
            mask[N-1,:]=1
            mask[:,N-1]=1
            mask[N-1,N-1]=0
            W = np.multiply(W,mask)
        elif topology == 'ring_center':
            mask = np.eye(N-1)
            mask = np.roll(mask,1,axis=1)
            mask = np.insert(mask,N-1,1,axis=0)
            mask = np.insert(mask,N-1,1,axis=1)
            mask[-1,-1]=0
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
