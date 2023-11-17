import os
from typing import Dict, Any

import torch
import wandb
import pytorch_lightning as pl
import numpy as np
from PIL import Image


class BBoxPlModel(pl.LightningModule):
    def __init__(self, model, lr, metrics):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = model.loss
        self.metrics = metrics


    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        imgs, gt_masks = batch
        pred_masks = self.model(imgs)
        loss = self.criterion(pred_masks, gt_masks)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        for metric in self.metrics:
            metric_val = metric(pred_masks, gt_masks)
            self.log(
                f"train_{metric.__name__}",
                metric_val,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return loss

    def _save_batch_masks(self, epoch_idx, imgs, pred_masks):
        output_dir = self.masks_debug_output / f"epoch_{epoch_idx}"
        os.makedirs(output_dir, exist_ok=True)
        for i, (img, mask) in enumerate(zip(imgs, pred_masks)):
            img = (img.cpu().numpy() * 255).astype(np.uint8)
            mask = mask.cpu().numpy().astype(np.uint8)
            img = Image.fromarray(np.moveaxis(img, 0, 2))
            mask = Image.fromarray(np.squeeze(mask))
            mask = mask.convert("RGB")
            img.save(output_dir / f"img_{i}.png")
            mask.save(output_dir / f"mask_{i}.png")
        print(f"Saved masks to {output_dir}")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred_masks = self.model(x)
        loss = self.criterion(pred_masks, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        for metric in self.metrics:
            metric_val = metric(pred_masks, y)
            self.log(
                f"val_{metric.__name__}",
                metric_val,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        # Only save for the first batch
        if batch_idx == 0:
            if self.trainer.sanity_checking:
                epoch_idx = "sanity_check"
            else:
                epoch_idx = self.current_epoch
            self._save_batch_masks(epoch_idx, x, pred_masks)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            # Higher wd = Reduce over-fitting but also reduce the capacity of the model
            # to make good preds
            # Lower wd = Increases the capacity of the model, increases over-fitting
            weight_decay=0.05,
            lr=self.lr,
        )
        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Save the wsis used for training and validation
        checkpoint["train_wsis"] = self.train_wsis
        checkpoint["val_wsis"] = self.val_wsis
        checkpoint["wand_train_url"] = wandb.run.get_url()
        super().on_save_checkpoint(checkpoint)


