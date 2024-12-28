import argparse
import os
import random
import yaml
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from funcodec.tasks.text2audio_generation import Text2AudioGenTask
from funcodec.schedulers.warmup_lr import WarmupLR
from funcodec.torch_utils.load_pretrained_model import load_pretrained_model

from _funcodec import init_sequence_iter_factory

from trainer.abs_trainer import Trainer
from utils import setup_logger
from utils import init
from utils import AttrDict


def setup_seed(seed, rank):
    SEED = int(seed) + rank
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return SEED


## ddp process
def setup(rank, world_size, backend, port=12355):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, args):
    args = AttrDict(**vars(args))
    l = setup_logger(args, rank)
    l.info("logging initialized succesully")
    l.info(args)
    l.info(f"rank {rank} of world_size {len(args.gpus)} started...")
    setup(rank, len(args.gpus), args.dist_backend)
    args.gpu = args.gpus[rank]
    torch.cuda.set_device(args.gpu)
    torch.cuda.empty_cache()
    setup_seed(args.seed, rank)
    l.info("setup model")
    ## load laura gpt model
    model: nn.Module = Text2AudioGenTask.build_model(args)
    model.cuda()
    for p in args.init_param:
        l.info(f"Loading pretrained params from {p}")
        load_pretrained_model(
            model=model,
            init_param=p,
            ignore_init_mismatch=True,
            # NOTE(kamo): "cuda" for torch.load always indicates cuda:0
            #   in PyTorch<=1.4
            map_location=f"cuda:{torch.cuda.current_device()}",
        )
    model = DDP(model, device_ids=[args.gpu])
    l.info(f"model {model} is intialized")
    ## optimizer
    optim = init(torch.optim, args.optim, model.parameters())
    ## scheduler
    assert args.scheduler == "warmuplr"
    scheduler = WarmupLR(optim, **args.scheduler_conf)
    l.info(f"scheduler {scheduler} and optim {optim} is initialized")
    ## setup dataloader
    ### Initialized iter factory
    train_iter = init_sequence_iter_factory(args, rank, "train")
    val_iter = init_sequence_iter_factory(args, rank, "valid")

    ## ckpt_dir
    ckpt_dir = os.path.basename(args.config).replace(".yaml", "")

    trainer = Trainer(
        model,
        train_iter,
        val_iter,
        optim,
        scheduler,
        config=args,
        ckpt_dir=f"./ckpt/{ckpt_dir}",
        rank=rank,
        logger=l,
    )
    l.info("starting training!")
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument("--gpus", default="0,1,2,3", type=str, help="gpus")
    parser.add_argument("--log", default="./log", type=str, help="Output of the log")
    parser.add_argument("--config", type=str, default=None, help="path to yaml config")
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, "r") as file:
            config = yaml.safe_load(file)
        for k, v in config.items():
            args.__setattr__(k, v)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.gpus = [int(i) for i in args.gpus.split(",")]
    args.ngpu = len(args.gpus)

    mp.spawn(main, args=(args,), nprocs=len(args.gpus), join=True)
    pass
