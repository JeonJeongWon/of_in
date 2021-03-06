import warnings
import os
import argparse
from dataclasses import dataclass

parser = argparse.ArgumentParser()

# system parameter
parser.add_argument("--num_sc",default=64,type=int,help="Number Of Subarriers")
parser.add_argument("--num_ch_paths",default=2,type=int,help="Number Of Channel paths")
parser.add_argument("--bps",default=2,type=int,help="Number Of Bits per Symbol")
parser.add_argument("--num_datas",default=200000,type=int,help="Number Of Data")
#parser.add_argument("--FFTsize",default=)

# channel parameter
parser.add_argument("--num_paths",default=2,type=int,help="Number Of Channel paths")
parser.add_argument("--snrdb",default=10,type=int,help="SNRdB")

## NN parameter
parser.add_argument("--num_nodes",default=2048,type=int,help='Number of Nodes in Layer')
parser.add_argument("--batch_size",default=128, type=int,help="Batch Size")
parser.add_argument("--epochs",default=10,type=int,help="Epochs")
parser.add_argument("--weight",default=0.01,type=float,help="Weights for loss fumction")
parser.add_argument("--lr",default=1e-4,type=float,help="Learning Rate")
parser.add_argument("--patience",default=10,type=int,help="EarlyStopping_patience")
parser.add_argument("--wait",default=0,type=int,help="EarlyStopping_wait")
parser.add_argument("--best_1",default=0.0,type=float,help="EarlyStopping_best")
parser.add_argument("--best_2",default=1.0,type=float,help="EarlyStopping_best")

args = parser.parse_args()

@dataclass
class system:
    num_sc = args.num_sc
    num_ch_paths = args.num_paths
    num_datas = args.num_datas
    bps = args.bps
sys = system
#----------------------------------------
@dataclass
class channel:
    num_paths = args.num_paths
    snrdb = args.snrdb
ch = channel
#----------------------------------------
@dataclass
class neuralnet:
    num_nodes = args.num_nodes
    batch_size = args.batch_size
    epochs = args.epochs
    weight = args.weight
    lr = args.lr
    patience = args.patience
    wait = args.wait
    best_1 = args.best_1
    best_2 = args.best_2

nn = neuralnet