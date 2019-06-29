# echo state network
echo state network用のプログラム  
train.pyを実行  

#### args一覧
##### for reservoir
- '--in_size', '-i', type=int,help='input size(default=1)', default=1  
- '--res_size', '-r', type=int,help='reservoir size(default=100)', default=100  
- '--out_size', '-o', type=int,help='output size(default=1)', default=1  
- '--con', '-c', type=float,help='connectivity range:0~1(default=0.05)', default=0.05  
- '--dataset', '-d', type=str,help='dataset name MG:Mackey_Glass(default)', default='MG'  
- '--data_len', '-l', type=int,help='dataset length(default=2000)', default=2000  
- '--folder', '-f', type=str,help='output folder name(default=nowtime)',  default=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  
- '--topology', '-t', type=str,help='topology:array (default:array)', default='random'  

##### for readout (for chainer)
- '--epoch', '-e', type=int, default=100, help='Number of sweeps over the dataset to train'
- '--batchsize', '-b', type=int, default=100, help='Number of images in each mini-batch'
- '--frequency', type=int, default=-1, help='Frequency of taking a snapshot'
- '--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)'
- '--resume', default='', help='Resume the training from snapshot'
- '--noplot', dest='plot', action='store_false', help='Disable PlotReport extension'
