import sys,os
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import model
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import yaml
import cv2
import ctools
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler
import argparse

def main(config):

    #  ===================>> Setup <<=================================

    dataloader = importlib.import_module("reader." + config.reader)

    torch.cuda.set_device(config.device) 
    cudnn.benchmark = True

    data = config.data
    save = config.save
    params = config.params

    print("===> Read data <===")

    if data.isFolder:
        data, _ = ctools.readfolder(data)

    dataset = dataloader.loader(
                    data,
                    params.batch_size, 
                    shuffle=True, 
                    num_workers=8
                )


    print("===> Model building <===")
    net = model.Model()
    net.train(); net.cuda()


    # Pretrain 
    pretrain = config.pretrain

    if pretrain.enable and pretrain.device:
        net.load_state_dict( 
                torch.load(
                    pretrain.path, 
                    map_location={f"cuda:{pretrain.device}": f"cuda:{config.device}"}
                )
            )
    elif pretrain.enable and not pretrain.device:
        net.load_state_dict(
                torch.load(pretrain.path)
                )


    print("===> optimizer building <===")
    optimizer = optim.Adam(
                    net.parameters(),
                    lr=params.lr, 
                    betas=(0.9,0.999)
                )
  
    scheduler = optim.lr_scheduler.StepLR( 
                    optimizer, 
                    step_size=params.decay_step, 
                    gamma=params.decay
                )

    if params.warmup:
        scheduler = GradualWarmupScheduler( 
                        optimizer, 
                        multiplier=1, 
                        total_epoch=params.warmup, 
                        after_scheduler=scheduler
                    )

    savepath = os.path.join(save.metapath, save.folder, f"checkpoint")

    if not os.path.exists(savepath):
        os.makedirs(savepath)
 
    # =====================================>> Training << ====================================
    print("===> Training <===")

    length = len(dataset); total = length * params.epoch
    timer = ctools.TimeCounter(total)


    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()


    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
        outfile.write(ctools.DictDumps(config) + '\n')

        for epoch in range(1, params.epoch+1):
            for i, (data, anno) in enumerate(dataset):

                # -------------- forward -------------
                for key in data:
                    if key != 'name': data[key] = data[key].cuda()

                anno = anno.cuda() 
                loss = net.loss(data, anno)

                # -------------- Backward ------------
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                rest = timer.step()/3600


                if i % 20 == 0:
                    log = f"[{epoch}/{params.epoch}]: " + \
                          f"[{i}/{length}] " +\
                          f"loss:{loss} " +\
                          f"lr:{ctools.GetLR(optimizer)} " +\
                          f"rest time:{rest:.2f}h"

                    print(log); outfile.write(log + "\n")
                    sys.stdout.flush(); outfile.flush()

            scheduler.step()

            if epoch % save.step == 0:
                torch.save(
                        net.state_dict(), 
                        os.path.join(
                            savepath, 
                            f"Iter_{epoch}_{save.model_name}.pt"
                            )
                        )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch Basic Model Training')

    parser.add_argument('-s', '--train', type=str,
                        help='The source config for training.')

    args = parser.parse_args()

    config = edict(yaml.load(open(args.train), Loader=yaml.FullLoader))

    print("=====================>> (Begin) Training params << =======================")
    print(ctools.DictDumps(config))
    print("=====================>> (End) Traning params << =======================")

    main(config.train)

