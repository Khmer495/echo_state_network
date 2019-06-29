# -*- coding: utf-8 -*-

import numpy as np
#import numpy.random as rd
#import numpy.linalg as LA
#import pickle
#import pandas as pd
import os
import argparse
import datasets
import reservoir
import readout
import datetime
import compare
import pickle
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_size','-i',type=int,help='input size(default=1)',default=1)
    parser.add_argument('--res_size','-r',type=int,help='reservoir size(default=100)',default=100)
    parser.add_argument('--out_size','-o',type=int,help='output size(default=1)',default=1)
    parser.add_argument('--con','-c',type=float,help='connectivity range:0~1(default=0.05)',default=0.05)
    parser.add_argument('--make',type=int,help='random weight make Yes:-1(default),No:0',default=-1)
    parser.add_argument('--dataset','-d',type=str,help='dataset name MG:Mackey_Glass(default)' ,default='MG')
    parser.add_argument('--data_len','-l',type=int,help='dataset length(default=2000)',default=2000)
    parser.add_argument('--folder','-f',type=str,help='output folder name(default=nowtime)',default=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    parser.add_argument('--topology','-t',type=str,help='topology:array (default:array)',default='random')

    #for chainer
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--frequency', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()

    args.folder = '{}'.format(args.folder)
    os.mkdir(args.folder)

    '''
    print('[args]')
    f = open('{}/args.csv'.format(args.folder),'w')
    for key,item in vars(args).items():
        print('{}:{}'.format(key,item))
        f.write('{}\t{}'.format(key,item))
    f.close()
    '''

# make_input_dataset
    print('make dataset')
    if args.dataset == 'MG':
        input_train = datasets.Mackey_Glass_equation(1.2, args.data_len, args.in_size)
        input_test = datasets.Mackey_Glass_equation(0.2, args.data_len, args.in_size)
    else:
        print('no dataset')
        sys.exit()

# set_reservoir
    print('set reservoir')
    if args.topology == 'random':
        rsv = reservoir.Random(args.folder,args.in_size,args.res_size,args.con)
    else:
        print('no topology')
        sys.exit()


# run_train_reservoir
    print('make x_train')
    x_train, flow_train = rsv.run(input_train, args.data_len)
    '''
    with open('{}/reservoir_flow_train.pickle'.format(args.folder), mode='wb') as f:
        pickle.dump(flow_train, f)
    '''

# run_test_reservoir
    print('make x_test')
    x_test, flow_test = rsv.run(input_test, args.data_len)
    '''
    with open('{}/reservoir_flow_test.pickle'.format(args.folder), mode='wb') as f:
        pickle.dump(flow_test, f)
    '''

    #np.savez('{}/dataset{}.npz'.format(args.folder,args.dataset),train_input=input_train,test_input=input_test,train_reservoir=x_train,test_reservoir=x_test)

# cut_begin_traindata and reshape for chainer model
    print('arrange datasets for chainer')
    cut = 100
    input_train_cut = input_train.reshape(args.data_len,args.in_size)[cut+1:-1]
    x_train_cut = x_train.reshape(args.data_len,args.res_size)[cut:-2]

    input_test_cut = input_test.reshape(args.data_len,args.in_size)[cut+1:-1]
    x_test_cut = x_test.reshape(args.data_len,args.res_size)[cut:-2]

    print('[learning]')
    readout.main(args,[x_train_cut,input_train_cut],[x_test_cut,input_test_cut])

    print('draw compare graph')
    snapshot = np.load('{}/snapshot.npz'.format(args.folder))
    Wout = snapshot['updater/model:main/predictor/l1/W']
    b = snapshot['updater/model:main/predictor/l1/b']
    compare.fig(input_train, x_train, Wout, b, args.folder, 'train')
    compare.fig(input_test, x_test, Wout, b, args.folder, 'test')

if __name__=="__main__":
    main()
