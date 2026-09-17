#! /bin/bash
python morphcell/main.py \
    --config-name=pretrain_cytodl_point \
    system.optimizer_cfg.epochs=300 \
    system.resume_ckpt_path=null \
    system.pretrained_ckpt_path=null \
    system.reconstructor.num_output_points=2025 \
    data.train_datasets.shapenet55_34.num_points=2025 \
    trainer.accelerator=gpu \
    trainer.devices=auto \
    trainer.strategy=ddp \
    trainer.accumulate_grad_batches=1 \
    data.batch_size=16 \
    logger.wandb.id=null \
    logger.wandb.name="pretrain_cytodl_point" \
    logger.wandb.mode="offline" \
    hydra.run.dir="outputs/pretrain/pretrain_cytodl_point"
