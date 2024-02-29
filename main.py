import os
import sys
import torch
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from model.MSTAGNN import Network

from trainer import Trainer
from utils.util import init_seed
from utils.dataloader import get_dataloader_pems
from utils.util import print_model_parameters
import warnings

from utils.metrics import MAE_torch

warnings.filterwarnings('ignore')

#*************************************************************************#


file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss
def init_model(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    print_model_parameters(model, only_num=False)

    return model
# Mode = 'train'
# DEBUG = 'True'
# DATASET = 'PEMSD3'      #PEMSD4 or PEMSD8
# DEVICE = 'cuda:0'
# MODEL = 'DDGCRN'

#parser
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default='PEMS08', type=str)
args.add_argument('--mode', default='train', type=str)
args.add_argument('--device', default='cuda:0', type=str, help='indices of GPUs')
args.add_argument('--debug', default='False', type=eval)
args.add_argument('--model', default='MGSTGNN', type=str)
args.add_argument('--cuda', default=True, type=bool)
args1 = args.parse_args()

#get configuration
config_file = './config/{}.conf'.format(args1.dataset)
#print('Read configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)

#data
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
args.add_argument('--in_steps', default=config['data']['in_steps'], type=int)
args.add_argument('--out_steps', default=config['data']['out_steps'], type=int)
args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
args.add_argument('--adj_norm', default=config['data']['adj_norm'], type=eval)
#model
args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)

args.add_argument('--num_input_dim', default=config['model']['num_input_dim'], type=int)


args.add_argument('--periods_embedding_dim', default=config['model']['periods_embedding_dim'], type=int)
args.add_argument('--weekend_embedding_dim', default=config['model']['weekend_embedding_dim'], type=int)

args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int)
args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
args.add_argument('--periods', default=config['model']['periods'], type=int)
args.add_argument('--weekend', default=config['model']['weekend'], type=int)
args.add_argument('--kernel', default=config['model']['kernel'], type=int)

#train
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--random', default=config['train']['random'], type=eval)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
args.add_argument('--real_value', default=config['train']['real_value'], type=eval, help = 'use real value for loss calculation')

#test
args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
#log
args.add_argument('--log_dir', default='./', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)
args = args.parse_args()

if args.random:
    args.seed = torch.randint(10000, (1,))
print(args)
init_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.set_device(int(args.device[5]))
else:
    args.device = 'cpu'

#init model
model = Network(args)
model = model.to(args.device)
model = init_model(model)

#load dataset
train_loader, val_loader, test_loader, scaler = get_dataloader_pems(args1.dataset,args.batch_size,
                            args.val_ratio,args.test_ratio,args.in_steps,args.out_steps)

#init loss function, optimizer
if args.loss_func == 'mask_mae':
    loss_generator = masked_mae_loss(scaler, mask_value=0.0)
elif args.loss_func == 'mae':
    loss_generator = torch.nn.L1Loss().to(args.device)
elif args.loss_func == 'mse':
    loss_generator = torch.nn.MSELoss().to(args.device)
elif args.loss_func == 'huber':
    loss_generator = torch.nn.HuberLoss().to(args.device)
else:
    raise ValueError
optimizer_G = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                             weight_decay=0, amsgrad=False)
#learning rate decay
lr_scheduler_G = None

if args.lr_decay:
    print('Applying learning rate decay.')
    lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
    lr_scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_G,
                                                        milestones=lr_decay_steps,
                                                        gamma=args.lr_decay_rate)
#config log path
current_time = datetime.now().strftime('%Y%m%d%H%M%S')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir,'exps/logs', args.dataset, current_time)
args.log_dir = log_dir

#start training
trainer = Trainer(args,
                  model,
                  train_loader,val_loader,test_loader,scaler,
                  loss_generator,
                  optimizer_G,
                  lr_scheduler_G
                  )
if args.mode == 'train':
    trainer.train()
elif args.mode == 'test':
    model.load_state_dict(torch.load('./trained/{}/best_model.pth'.format(args.dataset)))
    print("Load saved model")
    trainer.test(model, trainer.args, test_loader, scaler, trainer.logger)
else:
    raise ValueError
