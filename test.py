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
