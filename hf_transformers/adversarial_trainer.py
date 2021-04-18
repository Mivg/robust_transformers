# This trainer is based on the trainer as forked from version of transformers in hash 935e346959e81aa96a772c11d05734ea652932d1
# And changed the training process to include discrete adversarial training
# TODO - perhaps shoould copy the trainer code here as well?

import collections
from datetime import datetime
import math
import os
from time import time
from typing import Optional, Union, Dict, Any, Tuple
import pandas as pd

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch import nn
from transformers import Trainer, TrainerState, is_apex_available, PreTrainedModel, is_torch_tpu_available, WEIGHTS_NAME
from packaging import version
from transformers.integrations import hp_params
from transformers.trainer_utils import TrainOutput
from transformers.utils import logging

from attacks.analyze_robustness import RunningCounter
from hf_transformers.adv_training_args import AdversarialTrainingArguments
from hf_transformers.adv_utils import get_adversarial_inputs, split_batch, get_attacker
from common.utils import set_seed_everywhere

_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):

    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True


logger = logging.get_logger(__name__)


class AdversarialTrainer(Trainer):
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch, optimized for 🤗 Transformers.

    Args:
        model (:class:`~transformers.PreTrainedModel` or :obj:`torch.nn.Module`, `optional`):
            The model to train, evaluate or use for predictions. If not provided, a ``model_init`` must be passed.

            .. note::

                :class:`~transformers.Trainer` is optimized to work with the :class:`~transformers.PreTrainedModel`
                provided by the library. You can still use your own models defined as :obj:`torch.nn.Module` as long as
                they work the same way as the 🤗 Transformers models.
        args (:class:`~transformers.TrainingArguments`, `optional`):
            The arguments to tweak for training. Will default to a basic instance of
            :class:`~transformers.TrainingArguments` with the ``output_dir`` set to a directory named `tmp_trainer` in
            the current directory if not provided.
        data_collator (:obj:`DataCollator`, `optional`):
            The function to use to form a batch from a list of elements of :obj:`train_dataset` or :obj:`eval_dataset`.
            Will default to :func:`~transformers.default_data_collator` if no ``tokenizer`` is provided, an instance of
            :func:`~transformers.DataCollatorWithPadding` otherwise.
        train_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
            The dataset to use for training. If it is an :obj:`datasets.Dataset`, columns not accepted by the
            ``model.forward()`` method are automatically removed.
        eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
             The dataset to use for evaluation. If it is an :obj:`datasets.Dataset`, columns not accepted by the
             ``model.forward()`` method are automatically removed.
        tokenizer (:class:`PreTrainedTokenizerBase`, `optional`):
            The tokenizer used to preprocess the data. If provided, will be used to automatically pad the inputs the
            maximum length when batching inputs, and it will be saved along the model to make it easier to rerun an
            interrupted training or reuse the fine-tuned model.
        model_init (:obj:`Callable[[], PreTrainedModel]`, `optional`):
            A function that instantiates the model to be used. If provided, each call to
            :meth:`~transformers.Trainer.train` will start from a new instance of the model as given by this function.

            The function may have zero argument, or a single one containing the optuna/Ray Tune trial object, to be
            able to choose different architectures according to hyper parameters (such as layer count, sizes of inner
            layers, dropout probabilities etc).
        compute_metrics (:obj:`Callable[[EvalPrediction], Dict]`, `optional`):
            The function that will be used to compute metrics at evaluation. Must take a
            :class:`~transformers.EvalPrediction` and return a dictionary string to metric values.
        callbacks (List of :obj:`~transformers.TrainerCallback`, `optional`):
            A list of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in :doc:`here <callback>`.

            If you want to remove one of the default callbacks used, use the :meth:`Trainer.remove_callback` method.
        optimizers (:obj:`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR`, `optional`): A tuple
            containing the optimizer and the scheduler to use. Will default to an instance of
            :class:`~transformers.AdamW` on your model and a scheduler given by
            :func:`~transformers.get_linear_schedule_with_warmup` controlled by :obj:`args`.
    """
    def __init__(self, max_seq_len, adv_training_args: AdversarialTrainingArguments, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_seq_len = max_seq_len
        self.adv_args = adv_training_args
        if self.adv_args.adversarial_every != 1:
            raise NotImplementedError('There is currently a problem with the implementation of not exploring every time, '
                                      'so this cannot be used. use adversarial_every 1')

    @staticmethod
    def _update_timing(timings, key, t):
        timings[key].update(time() - t)
        return time()

    def train(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        """
        Main training entry point.

        Args:
            model_path (:obj:`str`, `optional`):
                Local path to the model if the model to train has been instantiated from a local path. If present,
                training will resume from the optimizer/scheduler states loaded here.
            trial (:obj:`optuna.Trial` or :obj:`Dict[str, Any]`, `optional`):
                The trial run or the hyperparameter dictionary for hyperparameter search.
        """
        # This might change the seed so needs to run first.
        print('Entered adversarial trainer train procedure')
        t0 = datetime.now()
        t = time()
        timings = collections.defaultdict(RunningCounter)

        self._hp_search_setup(trial)

        t = self._update_timing(timings, 'hp_setup', t)

        # Model re-init
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            set_seed_everywhere(self.args.seed)

            model = self.call_model_init(trial)

            if not self.args.model_parallel:
                self.model = model.to(self.args.device)

            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        t = self._update_timing(timings, 'model_init', t)

        # Keeping track whether we can can len() on the dataset or not
        train_dataset_is_sized = isinstance(self.train_dataset, collections.abc.Sized)

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        t = self._update_timing(timings, 'train dataloader', t)
        adv_train_dataloaders = self.get_adv_train_dataloaders()
        t = self._update_timing(timings, 'adv dataloaders', t)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        if train_dataset_is_sized:
            num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if self.args.max_steps > 0:
                max_steps = self.args.max_steps
                num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                    self.args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(self.args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(self.args.num_train_epochs)
        else:
            # see __init__. max_steps is set when the dataset has no __len__
            max_steps = self.args.max_steps
            num_train_epochs = 1
            num_update_steps_per_epoch = max_steps

        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(model_path)

        # Mixed precision training with apex (torch < 1.6)
        model = self.model
        if self.args.fp16 and _use_apex:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1 and not self.args.model_parallel:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=(
                    not getattr(model.config, "gradient_checkpointing", False)
                    if isinstance(model, PreTrainedModel)
                    else True
                ),
            )
        # find_unused_parameters breaks checkpointing as per
        # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021

        # Train!
        if is_torch_tpu_available():
            total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )

        num_examples = (
            self.num_examples(train_dataloader)
            if train_dataset_is_sized
            else total_train_batch_size * self.args.max_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0

        t = self._update_timing(timings, 'final prep', t)

        # Check if continuing training from a checkpoint
        if model_path and os.path.isfile(os.path.join(model_path, "trainer_state.json")):
            self.state = TrainerState.load_from_json(os.path.join(model_path, "trainer_state.json"))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not self.args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= self.args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not self.args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch."
                )
            t = self._update_timing(timings, 'load checkpoint', t)

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        self.state.trial_params = hp_params(trial) if trial is not None else None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(self.args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        self._total_flos = self.state.total_flos
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not self.args.ignore_data_skip:
            for epoch in range(epochs_trained):
                # We just need to begin an iteration to create the randomization of the sampler.
                for _ in train_dataloader:
                    break

        # self.adv_args.adversarial_every *= self.args.gradient_accumulation_steps  # This was not correct!!!!
        total_steps_taken = 0
        total_steps = len(train_dataloader) * self.args.num_train_epochs

        t = self._update_timing(timings, 'setters', t)

        attacker = None if self.adv_args.orig_lambda == 1. else get_attacker(self.max_seq_len, self.args.device,
                                                                             self.args.train_batch_size, self.adv_args, self.tokenizer)

        t = self._update_timing(timings, 'attackers init', t)

        # ------------------------------------------------------------------------------------------------------------------
        # ----------------------------- actually starting the training (iterating over epochs) -----------------------------
        # ------------------------------------------------------------------------------------------------------------------
        for epoch in range(epochs_trained, num_train_epochs):
            # print(f'entering epoch {epoch}/{epochs_trained}')
            t = time()
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(epoch_iterator) if train_dataset_is_sized else self.args.max_steps
            self.control = self.callback_handler.on_epoch_begin(self.args, self.state, self.control)

            adv_batch_counter = -1

            adv_epoch_iterators = None
            if self.adv_args.orig_lambda < 1. and self.adv_args.async_adv_batches:
                adv_epoch_iterators = [iter(atdl) for atdl in adv_train_dataloaders]

            t = self._update_timing(timings, 'prep epoch', t)

            # *************************************************************************************************
            # ************************************ Starting training steps ************************************
            # *************************************************************************************************
            for step, inputs in enumerate(epoch_iterator):
                # batch will contain train_batch_size samples (n_gpus*batch_per_gpu)

                # TODO - perhpas the following lines shouold be in the on_step_begins callback?
                total_steps_taken += 1
                # print('Starting batch:', torch.cuda.memory_allocated() / (2**20))
                adv_batch_counter += 1

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(self.args, self.state, self.control)

                if (
                    ((step + 1) % self.args.gradient_accumulation_steps != 0)
                    and self.args.local_rank != -1
                    and _use_ddp_no_sync
                ):
                    with model.no_sync():
                        # print(f'start training step')
                        # ------------------------>>>>> Perform training step <<<<<<------------------------
                        tr_loss += self.training_step(model, inputs)
                        t = self._update_timing(timings, 'train step', t)

                        # print(f'start adv training step')
                        # ------------------------>>>>> Perform Adversarial training step <<<<<<------------------------
                        tr_loss += self.adv_training_step(model, inputs, tr_loss, adv_epoch_iterators, adv_batch_counter,
                                                          step, attacker, timings)
                        t = self._update_timing(timings, 'adv train step', t)

                else:
                    # print(f'start training step')

                    # ------------------------>>>>> Perform training step <<<<<<------------------------
                    tr_loss += self.training_step(model, inputs)
                    t = self._update_timing(timings, 'train step', t)

                    # print(f'start adv training step')

                    # ------------------------>>>>> Perform Adversarial training step <<<<<<------------------------
                    tr_loss, adv_batch_counter = self.adv_training_step(model, inputs, tr_loss, adv_epoch_iterators, adv_batch_counter,
                                                                        step, attacker, timings)
                    t = self._update_timing(timings, 'adv train step', t)

                # print('out of steps')
                self._total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= self.args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    elif self.args.fp16 and _use_apex:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if is_torch_tpu_available():
                        xm.optimizer_step(self.optimizer)
                    elif self.args.fp16 and _use_native_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.lr_scheduler.step()
                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(self.args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

                    t = self._update_timing(timings, 'optimize step', t)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

            self.control = self.callback_handler.on_epoch_end(self.args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch)

            if self.args.tpu_metrics_debug or self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            logger.info(
                f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
            )
            if isinstance(self.model, PreTrainedModel):
                self.model = self.model.from_pretrained(self.state.best_model_checkpoint)
                if not self.args.model_parallel:
                    self.model = self.model.to(self.args.device)
            else:
                state_dict = torch.load(os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME))
                self.model.load_state_dict(state_dict)

        if self._total_flos is not None:
            self.store_flos()
            self.log({"total_flos": self.state.total_flos})

        self.control = self.callback_handler.on_train_end(self.args, self.state, self.control)
        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        return TrainOutput(self.state.global_step, self._total_loss_scalar / self.state.global_step)

    def get_adv_train_dataloaders(self):
        adv_train_dataloaders = None
        if self.adv_args.orig_lambda < 1. and self.adv_args.async_adv_batches:
            adv_train_dataloaders = [
                # Though there is no need to encode these batches we iterate over the train_dataset which is already encoded. Also, note that
                # there is no explicit see here but the random sampler will render each of the loader with a different permutation
                DataLoader(self.train_dataset,
                           sampler=self._get_train_sampler(),
                           batch_size=self.args.train_batch_size,
                           collate_fn=self.data_collator,
                           drop_last=self.args.dataloader_drop_last,
                           num_workers=self.args.dataloader_num_workers,
                           )
                for _ in range(self.adv_args.n_adv_loaders)
            ]
        return adv_train_dataloaders

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.args.fp16 and _use_native_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        loss = self.adv_args.orig_lambda * loss

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16 and _use_native_amp:
            self.scaler.scale(loss).backward()
        elif self.args.fp16 and _use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.detach()

    def adv_training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], tr_loss, adv_epoch_iterators,
                          adv_batch_counter: int, step: int, attacker, timings) -> Tuple[torch.Tensor, int]:
        # start attacking the inputs (attack all inputs in the batch)
        if self.adv_args.orig_lambda < 1. and adv_batch_counter % self.adv_args.adversarial_every == 0:
            # advance adversarial loaders or use current batch
            # TODO - perhphaps can avoid preparing them on gpu
            t = time()
            if not self.adv_args.async_adv_batches:
                adv_batches = self._prepare_inputs(inputs)  # for sync adv loaders we will exploit on each sample which is n_exploits*train_batch_size
            else:
                # with async loaders, each loader will create train_batch_size batches, so we would end up with
                # n_exploits*n_loaders*train_batch_size adv batches
                adv_batches = [self._prepare_inputs(next(aei)) for aei in adv_epoch_iterators]
                if len(adv_batches) > 1:
                    adv_batches = {k: torch.cat([b[k] for b in adv_batches]) for k in adv_batches[0].keys()}
                else:
                    adv_batches = adv_batches[0]

            if (step + 1) % self.args.gradient_accumulation_steps != 0:  # not the step that performs the update
                adv_batch_counter -= 1  # force next step to be adversarial as well
            # this is not
            model.eval()
            t = self._update_timing(timings, 'prep adv batch', t)
            with torch.no_grad():
                adv_inputs = get_adversarial_inputs(self.adv_args, adv_batches, model, self.tokenizer, attacker, self.args.eval_batch_size,
                                                    self._prepare_inputs, timings=timings)

                # # -------------------------------------------------------------------------------------------
                # print('\n' + ('*' * 75))
                # for sent, mask, lab in zip(adv_inputs['input_ids'].cpu(), adv_inputs['attention_mask'].cpu(), adv_inputs['labels'].cpu()):
                #     # validate:
                #     mask = mask.numpy()
                #     mask_end_ind = mask.sum()
                #     assert np.all(mask[:mask_end_ind] == 1)
                #     assert np.all(sent.numpy()[mask_end_ind:] == 0)
                #     assert sent[0].item() == 101
                #     assert sent[mask_end_ind - 1].item() == 102
                #     print(tokenizer.decode(sent.numpy(), skip_special_tokens=True), '\t', lab.item())
                # print('*' * 75)
                # # -------------------------------------------------------------------------------------------

            model.train()
            t = self._update_timing(timings, 'get adv batch', t)

            # print('Created adv inputs:', torch.cuda.memory_allocated() / (2**20))

            if adv_inputs is not None:
                # we may have more inputs here than allowed by the train_batch_size
                for mini_batch, weight in split_batch(adv_inputs, self.args.train_batch_size):
                    adv_loss = model(**mini_batch)[0]

                    # print('Forwarded it:', torch.cuda.memory_allocated() / (2**20))
                    if self.args.n_gpu > 1:
                        adv_loss = adv_loss.mean()  # mean() to average on multi-gpu parallel training

                    # Note that we already becakpropogagte the orignal loss as well as added it to the tr_loss counter, so here we only
                    # need to take care of the adv_loss
                    adv_loss = (1 - self.adv_args.orig_lambda) * adv_loss * weight  # the weight is the chunk of the original batch it plays

                    if self.args.gradient_accumulation_steps > 1:
                        adv_loss = adv_loss / self.args.gradient_accumulation_steps

                    if self.args.fp16 and _use_native_amp:
                        self.scaler.scale(adv_loss).backward()
                    elif self.args.fp16 and _use_apex:
                        with amp.scale_loss(adv_loss, self.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        adv_loss.backward()

                    return tr_loss+adv_loss.detach(), adv_batch_counter
            t = self._update_timing(timings, 'train on adv batch', t)

        else:
            return tr_loss, adv_batch_counter
