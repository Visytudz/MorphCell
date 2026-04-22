import logging

import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from morphcell.loss import ChamferLoss

logger = logging.getLogger(__name__)


class BaselinePretrain(pl.LightningModule):
    def __init__(
        self,
        reconstructor,
        transform=torch.nn.Identity(),
        optimizer_cfg=None,
        save_dir=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["reconstructor", "transform"])

        self.reconstructor = reconstructor
        self.extractor = reconstructor.encoder
        self.decoder = reconstructor.decoder
        self.transform = transform
        self.loss_fn = ChamferLoss()
        self.optimizer_cfg = optimizer_cfg
        self.save_dir = save_dir

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.optimizer_cfg.lr,
            weight_decay=self.optimizer_cfg.weight_decay,
        )

        warmup_epochs = self.optimizer_cfg.warmup_epochs
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=self.optimizer_cfg.warmup_lr_init / self.optimizer_cfg.lr,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )

        cosine_epochs = self.optimizer_cfg.epochs - warmup_epochs
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=self.optimizer_cfg.min_lr,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def load_pretrained_weights(self, checkpoint_path):
        if not checkpoint_path:
            logger.info("No checkpoint path provided, using random initialization.")
            return

        logger.info(f"Loading pretrained weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        logger.info(
            f"✓ Loaded {len(state_dict) - len(unexpected)} keys from checkpoint"
        )

        if missing:
            logger.info(
                f"Missing {len(missing)} keys (will use random initialization):"
            )
            for key in missing:
                logger.info(f"  - {key}")

        if unexpected:
            logger.info(f"Unexpected {len(unexpected)} keys (ignored):")
            for key in unexpected:
                logger.info(f"  - {key}")

    def training_step(self, batch, batch_idx):
        pts = batch["points"]
        pts = self.transform(pts)
        reconstruction = self.reconstructor(pts)
        loss_recon = self.loss_fn(pts, reconstruction)

        self.log_dict(
            {
                "train/loss": loss_recon,
                "train/loss_recon": loss_recon,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=pts.size(0),
        )

        return loss_recon
