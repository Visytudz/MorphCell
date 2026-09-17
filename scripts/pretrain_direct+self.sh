#! /bin/bash
python morphcell/main.py \
    --config-name=pretrain_direct+self \
    system.optimizer_cfg.epochs=300 \
    system.resume_ckpt_path=null \
    system.pretrained_ckpt_path=null \
    trainer.accelerator=gpu \
    trainer.devices=auto \
    trainer.strategy=ddp \
    trainer.accumulate_grad_batches=1 \
    data.batch_size=16 \
    logger.wandb.id=null \
    logger.wandb.name="pretrain_direct+self" \
    logger.wandb.mode="offline" \
    hydra.run.dir="outputs/pretrain/pretrain_direct+self"
