import os
import torch
import random
import numpy as np
import logging
from datetime import datetime
from utils.metrics import MAE_torch

def get_device(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device("cuda", args.gpu_id)
    else:
        device = torch.device("cpu")
    init_seed(args.seed) # reproductibility

    return device

def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_model_parameters(model, only_num = True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')

def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae

    return loss

def get_logger(root, name=None, debug=True):
    # when debug is true, show DEBUG and INFO in screen
    # when debug is false, show DEBUG in file and info in both screen&file
    # INFO will always be in screen
    # create a logger
    logger = logging.getLogger(name)
    # critical > error > warning > info > debug > notset
    logger.setLevel(logging.DEBUG)

    # define the formate
    formatter = logging.Formatter('%(asctime)s: %(message)s', "%Y-%m-%d %H:%M")
    # create another handler for output log to console
    console_handler = logging.StreamHandler()
    if debug:
        console_handler.setLevel(logging.DEBUG)
    else:
        console_handler.setLevel(logging.INFO)
        # create a handler for write log to file
        logfile = os.path.join(root, 'run.log')
        print('Creat Log File in: ', logfile)
        file_handler = logging.FileHandler(logfile, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # add Handler to logger
    logger.addHandler(console_handler)
    if not debug:
        logger.addHandler(file_handler)

    return logger

def generate_weekend(DATASET):
    day_dict = {
        'PEMS03':'2018-09-01',
        'PEMS04':'2018-01-01',
        'PEMS07':'2017-05-01',
        'PEMS08':'2016-07-01'
    }
    # get basic flow data
    flow_data = np.load('../dataset/{}/{}.npz'.format(DATASET,DATASET))['data']
    flow_data = flow_data[...,:1]
    print(flow_data.shape)
    time_stamps,num_nodes,dim = flow_data.shape
    start_date = datetime.strptime(day_dict[DATASET], "%Y-%m-%d")
    print('the start date of dataset: ', start_date)

    res = np.zeros((time_stamps,num_nodes,1))
    current_date = start_date
    for i in range(time_stamps//288):
        date_info = np.zeros((288,num_nodes))
        if current_date.weekday()>=5:
            date_info = np.ones((288,num_nodes))
            print("This is a weekend! ", current_date)

        res[i*288:(i+1)*288,:,0] = date_info

        current_date = current_date + dt.timedelta(days=1)
