import os
import gc; gc.enable()
import torch
import random
import warnings; warnings.filterwarnings(action='ignore')
import torchvision
import torchmetrics
import numpy as np
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from argparse import ArgumentParser
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

def build_args():
    parser = ArgumentParser()
    # example
    # parser.add_argument('--Something', default=1, type=int)
    args = parser.parse_args()

    return args

def build_dataloader(args):
    # print(args.dataset_name)
    train_loader, val_loader, test_loader = None, None, None
    return train_loader, val_loader, test_loader

class pl_model(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config_ = config
#         self.loss_fct = torch.nn.CrossEntropyLoss()
#         self.loss_fct = torch.nn.MSELoss()
        
        if self.config_.use_pre_train:
            self.model.load_state_dict({k.split('model.')[-1]:v for k,v in torch.load(self.config_.pre_train_weight_path)['state_dict'].items()})
            # PL ckpt savefile
            self.classifier = nn.Linear(self.config_.pre_train_weight_class, self.config_.num_classes)
            # classifier Layer
            print('use_pre_train')
        if self.config_.load_weight:
            self.model.load_state_dict({k.split('model.')[-1]:v for k,v in torch.load(self.config_.load_weight_path)['state_dict'].items()})
            print('load_weight')
            
    def forward(self, x):
        if self.config_.use_pre_train:
            features = self.model(x)
            return self.classifier(features)
        else:
            return self.model(x)


    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        output = self(x)
        loss = self.loss_fct(output, y)
#         acc = (output.argmax(dim=1) == y).float().mean()
        
        self.log("train_loss", loss, on_step = True, on_epoch = True, logger = True)
#         self.log("train_acc", acc, on_step = True, on_epoch = True, logger = True)
        return loss
    
    def validation_step (self, val_batch, batch_idx):
        x, y = val_batch
        output = self(x)
        loss = self.loss_fct(output, y)
        acc = (output.argmax(dim=1) == y).float().mean()
        
        self.log('val_loss', loss, on_step = True, on_epoch = True, logger = True)
#         self.log("val_acc", acc, on_step = True, on_epoch = True, logger = True)
        
    def configure_optimizers(self):
        if self.config_.optim == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.config_.learning_rate,
                                        momentum=0.9,
                                       )
        elif self.config_.optim == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), 
                                          lr=self.config_.learning_rate,
                                         )
        elif self.config_.optim == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=self.config_.learning_rate,
                                         )
        print(self.config_.optim)
        return [optimizer]
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                          milestones=[100,150],
#                                                          gamma=0.1)
#         return [optimizer], [scheduler]

    
class Custom_Model(nn.Module):
    def __init__(self, model_config, **kwargs):
        super().__init__()
        self.Layer = nn.Linear(10, 1)
        
    def forward(self, x):
        out_put = self.Layer(x)
        return out_put
    
def cli_main():
    args = build_args()
    
    train_loader, val_loader, test_loader = build_dr(args)
    custom_model = Custom_Model(**args)

    model = pl_model(custom_model, args)
    
    wandb_logger = WandbLogger(name=args.dataset_name + '_' + args.etc, project=args.project_name)

    if args.pre_train:
        callbacks = [
            ModelCheckpoint(
                dirpath = os.path.join('.','checkpoints', args.project_name),
                save_last = True,
                # filename = '{epoch:03d}-{train_loss:.3f}',
                # save_top_k = args.save_top_k,
                # monitor = 'val_loss',
            )
        ]
    else:
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=args.ealry_stop_round,
            )
        ]
    
    strategy = 'ddp' if args.gpu_numb > 1 else None
    enable_checkpointing = True if args.pre_train else False
    
    trainer = pl.Trainer(
        enable_checkpointing=enable_checkpointing,
        enable_progress_bar=args.prog_bar,
        max_epochs=args.epochs, 
        callbacks=callbacks,
        gpus = args.gpu_numb,
        logger=wandb_logger,
        strategy=strategy,
    )
    
    if args.pre_train:
        trainer.fit(model, train_loader)
    else:
        trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    cli_main()
