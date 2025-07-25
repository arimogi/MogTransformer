import os
import torch
import numpy as np
from models import MaelNet, KBJNet, DCDetector, AnomalyTransformer, FRNet, iTransformer

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            "MaelNet"    : MaelNet,            
            "KBJNet"     : KBJNet,
            "DCDetector" : DCDetector,
            "AnomalyTransformer":AnomalyTransformer,
            "FRNet"      : FRNet,
            "iTransformer" : iTransformer,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.slow_model = self._build_slow_model().to(self.device) if self._build_slow_model() else None
        self.URT = self._build_urt().to(self.device) if self._build_urt() else None
  
    def _build_model(self):
        raise NotImplementedError
    # For MANTRA
    def _build_slow_model(self):
        return None
    # For MANTRA
    def _build_urt(self):
        return None
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self, setting):
        pass