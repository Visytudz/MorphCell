#! /bin/bash
python morphcell/main.py \
    --config-name=dfn_dgcnn \
    system.optimizer_cfg.epochs=300 \
    system.resume_ckpt_path=null \
    system.pretrained_ckpt_path=null \
    system.reconstructor.encoder_type=dgcnn \
    trainer.accelerator=gpu \
    trainer.devices=auto \
    trainer.strategy=ddp \
    trainer.accumulate_grad_batches=1 \
    data.batch_size=16 \
    logger.wandb.id=null \
    logger.wandb.name="dfn_dgcnn" \
    logger.wandb.mode="offline" \
    hydra.run.dir="outputs/pretrain/dfn_dgcnn"
