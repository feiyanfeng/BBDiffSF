import sys
import os
import traceback

import torch
from trainer import Trainer
from tools.utils import task_wrapper
import hydra

from tools.send_email import send_email
from datetime import datetime

@task_wrapper
def train(args, logger):
    try:
        if args.mode=='train':
            runner = Trainer(args, logger)
            runner.train()
        elif args.mode=='test':
            runner = Trainer(args, logger)
            for i in range(args.test.ensemble_num):
                runner.test()
        else:
            raise NotImplementedError(f'{args.mode} not matched')
        logger.info('ending...\n\n\n')
        
    except Exception:
        # print(traceback.format_exc())
        logger.error(traceback.format_exc())
        # logger.info(f"{args.mode=='train'} , {'runner' in locals()} , {hasattr(runner, 'epoch')}")
        if args.mode=='train' and 'runner' in locals() and hasattr(runner, 'epoch'):  # 全局：globals()
            states = [
                    runner.model.state_dict(),
                    runner.optimizer.state_dict(),
                    runner.epoch
                ]
            if args.model.ema:
                states.append(runner.ema_helper.state_dict())
            torch.save(states, os.path.join(args.training.weigth_path, f"{args.name}_exception.pth"))
        return traceback.format_exc()
    
    return 0

@hydra.main(version_base="1.2", config_path="configs", config_name="main.yaml")
def main(cfg):
    time1 = datetime.now()
    feadback = train(cfg)
    time2 = datetime.now()
    send_email(cfg.name, time2-time1, feadback)
if __name__ == '__main__':
    sys.exit(main())
