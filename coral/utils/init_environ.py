# import numpy as np
# import random
import torch
import torch.distributed as dist

import os
import time
from pathlib import Path
import logging
import subprocess

def init_environ(cfg):

    
    # init distributed parallel
    if cfg.ddp.launcher == 'slurm':
        # one or multiple gpus
        _init_dist_slurm('nccl', cfg, cfg.ddp.port)
    elif cfg.ddp.launcher == 'pytorch':
        _init_dist_pytorch('nccl', cfg)
    else:
        # one gpu
        cfg.world_size = 1
        cfg.gpu_id = 0
        cfg.rank = 0
        cfg.distributed = False
        return 
    # else:
    #     raise NotImplementedError(f'launcher {cfg.launcher} has not been implemented.')
    cfg.distributed = True 
    torch.distributed.barrier()
    setup_for_distributed(cfg.rank == 0)

    # build work dir
    # exp_name = cfg.name # or config name
    # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # # work_dir = os.path.join('./exp', exp_name, timestamp+'_'+cfg.sub_name)
    # work_dir = os.path.join(cfg.exp_dir, exp_name, timestamp+'_'+cfg.sub_name)
    # if cfg.rank == 0:
    #     Path(work_dir).mkdir(parents=True, exist_ok=True)  
    # cfg.work_dir = work_dir      
    
    # # create logger
    # log_file = os.path.join(work_dir, 'log.txt')
    # logger = get_logger('search', log_file)
    # cfg.log_file = log_file
    # # set random seed
    # if cfg.seed is not None:
    #     set_random_seed(cfg.seed, deterministic=False, use_rank_shift=False)
    #     logger.info(f'set random seed to {cfg.seed}')
    # return logger

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def _init_dist_pytorch(backend, cfg):
    cfg.rank = int(os.environ['RANK'])
    # os.environ['LOCAL_RANK']
    cfg.world_size = int(os.environ['WORLD_SIZE'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(cfg.rank % num_gpus)
    
    cfg.gpu_id = cfg.rank % num_gpus
    dist.init_process_group(backend=backend)
    print(f'Distributed training on {cfg.rank}/{cfg.world_size}')
        
def _init_dist_slurm(backend, cfg, port=None):
    """Initialize slurm distributed training environment.
    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.
    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    # print(proc_id, num_gpus)
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)
    cfg.world_size = ntasks
    cfg.gpu_id = proc_id % num_gpus
    cfg.rank = proc_id

    dist.init_process_group(backend=backend)
    print(f'Distributed training on {proc_id}/{ntasks}')


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True
    logger.propagate = False

    return logger