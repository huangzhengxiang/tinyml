# Ultralytics YOLO 🚀, AGPL-3.0 license
from copy import copy, deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.ao.quantization.qconfig import QConfig, HistogramObserver
from torch.ao.quantization.observer import PerChannelMinMaxObserver
import time
from tqdm import tqdm
from datetime import datetime

from ultralytics.nn.tasks import DetectionModel, QuantDetectionModel
from ultralytics.yolo import v8
from ultralytics.yolo.data import build_dataloader, build_yolo_dataset
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.yolo.utils.loss import BboxLoss
from ultralytics.yolo.utils.ops import xywh2xyxy
from ultralytics.yolo.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.yolo.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors
from ultralytics.yolo.utils.torch_utils import de_parallel, torch_distributed_zero_first
from ultralytics.yolo.utils import (DEFAULT_CFG, LOGGER, ONLINE, RANK, ROOT, SETTINGS, TQDM_BAR_FORMAT, __version__,
                                    callbacks, clean_url, colorstr, emojis, yaml_save)
from ultralytics.nn.modules import Conv, DWConv
from ultralytics.nn.quantize.custom_quantized_format import QuantizedConv2d, get_effective_scale
from ultralytics.nn.quantize.quantized_ops import to_np, to_pt, USE_FP_SCALE


# BaseTrainer python usage
class DetectionTrainer(BaseTrainer):

    def build_dataset(self, img_path, mode='train', batch=None):
        """Build YOLO Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

    def get_dataloader(self, dataset_path, batch_size, rank=0, mode='train'):
        """TODO: manage splits differently."""
        # Calculate stride - check if model is initialized
        if self.args.v5loader:
            LOGGER.warning("WARNING ⚠️ 'v5loader' feature is deprecated and will be removed soon. You can train using "
                           'the default YOLOv8 dataloader instead, no argument is needed.')
            gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
            return create_dataloader(path=dataset_path,
                                     imgsz=self.args.imgsz,
                                     batch_size=batch_size,
                                     stride=gs,
                                     hyp=vars(self.args),
                                     augment=mode == 'train',
                                     cache=self.args.cache,
                                     pad=0 if mode == 'train' else 0.5,
                                     rect=self.args.rect or mode == 'val',
                                     rank=rank,
                                     workers=self.args.workers,
                                     close_mosaic=self.args.close_mosaic != 0,
                                     prefix=colorstr(f'{mode}: '),
                                     shuffle=mode == 'train',
                                     seed=self.args.seed)[0]
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        dataloader = build_dataloader(dataset, batch_size, workers, shuffle, rank)
        return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        """nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return v8.detect.DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def criterion(self, preds, batch):
        """Compute loss for YOLO prediction and ground-truth."""
        if not hasattr(self, 'compute_loss'):
            self.compute_loss = Loss(de_parallel(self.model))
        return self.compute_loss(preds, batch)

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(images=batch['img'],
                    batch_idx=batch['batch_idx'],
                    cls=batch['cls'].squeeze(-1),
                    bboxes=batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg')

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir)

# BaseTrainer python usage
class QuantDetectionTrainer(BaseTrainer):

    def build_dataset(self, img_path, mode='train', batch=None):
        """Build YOLO Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

    def get_dataloader(self, dataset_path, batch_size, rank=0, mode='train'):
        """TODO: manage splits differently."""
        # Calculate stride - check if model is initialized
        if self.args.v5loader:
            LOGGER.warning("WARNING ⚠️ 'v5loader' feature is deprecated and will be removed soon. You can train using "
                           'the default YOLOv8 dataloader instead, no argument is needed.')
            gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
            return create_dataloader(path=dataset_path,
                                     imgsz=self.args.imgsz,
                                     batch_size=batch_size,
                                     stride=gs,
                                     hyp=vars(self.args),
                                     augment=mode == 'train',
                                     cache=self.args.cache,
                                     pad=0 if mode == 'train' else 0.5,
                                     rect=self.args.rect or mode == 'val',
                                     rank=rank,
                                     workers=self.args.workers,
                                     close_mosaic=self.args.close_mosaic != 0,
                                     prefix=colorstr(f'{mode}: '),
                                     shuffle=mode == 'train',
                                     seed=self.args.seed)[0]
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        dataloader = build_dataloader(dataset, batch_size, workers, shuffle, rank)
        return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        """nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = QuantDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return v8.detect.DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def criterion(self, preds, batch):
        """Compute loss for YOLO prediction and ground-truth."""
        if not hasattr(self, 'compute_loss'):
            self.compute_loss = Loss(de_parallel(self.model))
        return self.compute_loss(preds, batch)

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(images=batch['img'],
                    batch_idx=batch['batch_idx'],
                    cls=batch['cls'].squeeze(-1),
                    bboxes=batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg')

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir)

    @torch.no_grad()
    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        assert world_size == 1           
        assert self.epochs == 1 
        self.ptq_ckpt="./ptq.pth"
        self.device = torch.device("cpu")
        self._setup_train(world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.epochs  # predefine for resume fully trained model edge cases
        self.model.fuse(fuse_all=True)
        self.model.eval()
        self.model.qconfig = QConfig(activation=HistogramObserver.with_args(dtype=torch.qint8,reduce_range=False),
                                weight=PerChannelMinMaxObserver.with_args(
                                    dtype=torch.qint8, qscheme=torch.per_channel_affine
                            ))
        # self.model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        print(self.model.qconfig)
        self.model = torch.quantization.prepare(self.model)
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.lr = self.lr = {'lr/pg': 0.}  # for loggers
            self.run_callbacks('on_train_epoch_start')
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)
                self.train_loader.reset()

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')

                batch = self.preprocess_batch(batch)
                preds = self.model(batch['img'])
                self.loss, self.loss_items = self.criterion(preds, batch)
                if RANK != -1:
                    self.loss *= world_size
                self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                    else self.loss_items

                # Log
                mem = f'{0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')

                self.run_callbacks('on_train_batch_end')
            
            self.model = torch.quantization.convert(self.model)
            print(self.model)
            state_dict = self.model.state_dict()
            print(len(state_dict))
            ptq_dict = {}
            for j, name in enumerate(state_dict):
                if hasattr(state_dict[name],"dtype"):
                    if state_dict[name].dtype == torch.qint8 or state_dict[name].dtype == torch.qint32:
                        # print(name,state_dict[name].int_repr())
                        ptq_dict[name]=state_dict[name].int_repr().numpy()
                        ptq_dict[name+".scales"]=state_dict[name].q_per_channel_scales().numpy()
                    else:
                        # print(name,state_dict[name])
                        ptq_dict[name]=state_dict[name].numpy()
                else:
                    # print(name,state_dict[name])
                    ptq_dict[name]=state_dict[name]
            torch.save(ptq_dict,self.ptq_ckpt)
            self.run_callbacks('on_train_epoch_end')

            if RANK in (-1, 0):

                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val or final_epoch:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks('on_model_save')

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        self.run_callbacks('teardown')

    def save_model(self):
        """Save model checkpoints based on various conditions."""
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)),
            'ema': deepcopy(self.ema.ema),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),  # save as dict
            'date': datetime.now().isoformat(),
            'version': __version__}

        # Save last, best and delete
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
        del ckpt

# BaseTrainer python usage
class QASDetectionTrainer(BaseTrainer):

    def build_dataset(self, img_path, mode='train', batch=None):
        """Build YOLO Dataset

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

    def get_dataloader(self, dataset_path, batch_size, rank=0, mode='train'):
        """TODO: manage splits differently."""
        # Calculate stride - check if model is initialized
        if self.args.v5loader:
            LOGGER.warning("WARNING ⚠️ 'v5loader' feature is deprecated and will be removed soon. You can train using "
                           'the default YOLOv8 dataloader instead, no argument is needed.')
            gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
            return create_dataloader(path=dataset_path,
                                     imgsz=self.args.imgsz,
                                     batch_size=batch_size,
                                     stride=gs,
                                     hyp=vars(self.args),
                                     augment=mode == 'train',
                                     cache=self.args.cache,
                                     pad=0 if mode == 'train' else 0.5,
                                     rect=self.args.rect or mode == 'val',
                                     rank=rank,
                                     workers=self.args.workers,
                                     close_mosaic=self.args.close_mosaic != 0,
                                     prefix=colorstr(f'{mode}: '),
                                     shuffle=mode == 'train',
                                     seed=self.args.seed)[0]
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        dataloader = build_dataloader(dataset, batch_size, workers, shuffle, rank)
        return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        """nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = QuantDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return v8.detect.DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def criterion(self, preds, batch):
        """Compute loss for YOLO prediction and ground-truth."""
        if not hasattr(self, 'compute_loss'):
            self.compute_loss = Loss(de_parallel(self.model))
        return self.compute_loss(preds, batch)

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor
        """
        # Not needed for classification but necessary for segmentation & detection
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(images=batch['img'],
                    batch_idx=batch['batch_idx'],
                    cls=batch['cls'].squeeze(-1),
                    bboxes=batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg')

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir)


    def generateQuantConv(self, ori_model: nn.Conv2d, new_dict: dict):
        kwargs = {
            'zero_x': to_pt(new_dict['x_zero']),
            'zero_w': to_pt(0),
            'zero_y': to_pt(new_dict['y_zero']),
        }
        effective_scale = get_effective_scale(new_dict['x_scale'], new_dict['w_scales'],
                                              new_dict['y_scale'])
        kwargs['effective_scale'] = to_pt(effective_scale)
        newConv = QuantizedConv2d(
            ori_model.in_channels, ori_model.out_channels, ori_model.kernel_size,
                padding=ori_model.padding, stride=ori_model.stride,
                groups=ori_model.groups, w_bit=8, a_bit=8
        )
        newConv.weight.data = to_pt(new_dict['weight'])
        newConv.bias.data = to_pt(new_dict['bias'])
        # Note that these parameters are added for convenience, not actually needed
        newConv.x_scale = new_dict['x_scale']
        newConv.y_scale = new_dict['y_scale']
        return newConv

    def _do_train(self, world_size=1):
        """Train completed, evaluate and plot if specified by arguments."""
        assert world_size == 1 
        self.ptq_ckpt = "./ptq_weight.pth"
        self.device = torch.device("cpu")
        self._setup_train(world_size)

        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100)  # number of warmup iterations
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        ptq_model = torch.load(self.ptq_ckpt)
        key_list = list(ptq_model.keys())
        cur_key = 0
        Conv.is_qas = True
        for module_name,m in self.model.named_children():
            if isinstance(m, nn.Conv2d):
                self.model.add_module(module_name,self.generateQuantConv(m,ptq_model[module_name]))
                print(module_name)
                
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)):
                while key_list[cur_key].split('.')[-1] != "conv":
                    cur_key += 1
                    continue
                m.update(ptq_model[key_list[cur_key]])
                
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.epochs  # predefine for resume fully trained model edge cases
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)
                self.train_loader.reset()

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = tqdm(enumerate(self.train_loader), total=nb, bar_format=TQDM_BAR_FORMAT)
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    preds = self.model(batch['img'])
                    self.loss, self.loss_items = self.criterion(preds, batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')

            if RANK in (-1, 0):

                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val or final_epoch:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')

    def save_model(self):
        """Save model checkpoints based on various conditions."""
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)),
            'ema': deepcopy(self.ema.ema),
            'updates': self.ema.updates,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),  # save as dict
            'date': datetime.now().isoformat(),
            'version': __version__}

        # Save last, best and delete
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
        if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
            torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
        del ckpt

# Criterion class for computing training losses
class Loss:

    def __init__(self, model):  # model must be de-paralleled

        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize YOLO model given training data and device."""
    model = cfg.model or 'yolov8n.pt'
    data = cfg.data or 'coco128.yaml'  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**args)
    else:
        trainer = DetectionTrainer(overrides=args)
        trainer.train()


if __name__ == '__main__':
    train()
