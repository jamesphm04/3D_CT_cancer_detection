import sys
import argparse
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from dsets import LunaDataset
from torch.utils.data import DataLoader



class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
            
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int                    
        )
        
        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=32,
            type=int,
        )
        
        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )
        
        self.cli_args = parser.parse_args(sys_argv)
        
        # self.use_cuda = torch.cuda.is_available()
        # self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        self.use_cuda = False
        self.device = torch.device('cpu')
        
        self.model = self.initModel()
        self.optimizer = self.initOptimizer()
        
    def initModel(self):
        model = LunaModel()
        if self.use_cuda:
            # log.info(f'Using CUDA; {torch.cuda.device_count()} devices')
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            
            model = model.to(self.device)
            
        return model
    
    def initOptimizer(self):
        return SGD(self.model.parameters(), lr=0.001, momentum=0.99)
        # return Adam(self.model.parameters())
        
    def initTrainDl(self):
        train_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=False
        )
        
        batch_size = self.cli_args.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
            
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )
        
        return train_dl
    
    def initValDl(self):
        val_ds = LunaDataset(
            val_stride=10,
            isValSet_bool=True
        )
        
        batch_size = self.cli_args.batch_size
        
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()
            
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=self.cli_args.num_workers,
            pin_memory=self.use_cuda
        )
        
        return val_dl
        
    def main(self):
        # log.info(f'Starting {type(self).__name__}, {self.cli_args}')
        train_dl = self.initTrainDl()
        val_dl = self.initValDl()
        
        
    
    