# from torch.optim.lr_scheduler import
import logging
import os
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch.nn.functional import softmax as torch_softmax
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

from .classifier import RoomModel
from .dataset import TorchDataset
from .utils import get_preprocessor


class TrainerUtils:
    def __init__(
        self,
        device='cpu',
        tensorboard_dir: Optional[str] = None,
        experiment_tag: Optional[str] = None,
        run_in_notebook=True,
        model_checkpoint_path: str = '/app/data/models/',
    ):
        self.device: str = device
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        self._global_steps_num = 0
        self.tensorboard: Optional[SummaryWriter] = None
        if tensorboard_dir is not None:
            if experiment_tag is not None:
                tensorboard_dir = os.path.join(tensorboard_dir, experiment_tag)
            self.tensorboard = SummaryWriter(tensorboard_dir)
        self.run_in_notebook = run_in_notebook
        self.best_accuracy: float = 0
        self.best_epoch: int = -1
        self.accuracies: List[float] = []
        self.checkpoint_path: str = os.path.join(model_checkpoint_path, f'model_{experiment_tag}')

    def get_progress_bar(self, epoch_len):
        if self.run_in_notebook:
            pbar = tqdm_notebook(total=epoch_len)
        else:
            pbar = tqdm(total=epoch_len)

        return pbar

    def log_metrics_loss(self, loss, name, step):
        self.tensorboard.add_scalar(f'Loss/{name}', loss, step)

    def log_metrics_accuracy(self, accuracy, name, step):
        self.tensorboard.add_scalar(f'Accuracy/{name}', accuracy, step)

    def do_checkpoint(
        self,
        epoch: int,
        accuracy: float,
        model,
        accuracy_df: pd.DataFrame = None
    ):
        self.accuracies.append(accuracy)

        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.best_epoch = epoch

            model_state = {
                'epoch': epoch,
                'accuracy': accuracy,
            }
            model.save_pretrained(self.checkpoint_path, model_state)

    def forward_step(self, batch, model) -> Tuple[torch.Tensor, float, list, list, torch.Tensor]:
        labels, sample_weights = batch[1], batch[2]
        logits = model(batch[0])

        loss_batch = self.loss_fn(logits, labels)
        if sample_weights is not None:
            loss_batch = loss_batch * sample_weights

        loss = loss_batch.mean()

        # loss = self.loss_fn(logits, labels)
        logits = logits.detach().cpu()
        labels_predicted = logits.argmax(dim=1).tolist()
        labels = labels.detach().cpu().tolist()

        accuracy = accuracy_score(labels, labels_predicted)

        return loss, accuracy, labels_predicted, labels, logits

    def training_step(
        self, model, batch: List[torch.Tensor], optimizer, scheduler, grad_clipping_norm
    ):
        model.train()
        batch[0] = batch[0].to(self.device)
        batch[1] = batch[1].to(self.device)
        batch[2] = batch[2].to(self.device)
        optimizer.zero_grad()

        loss, accuracy, labels_predicted, labels, logits = self.forward_step(batch, model)

        loss.backward()
        if grad_clipping_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        return loss.item(), accuracy

    def test_step(
        self, model, batch: List[torch.Tensor]
    ) -> Tuple[float, float, list, list, torch.Tensor]:
        model.eval()
        batch[0] = batch[0].to(self.device)
        batch[1] = batch[1].to(self.device)
        batch[2] = batch[2].to(self.device)
        with torch.no_grad():
            loss, accuracy, labels_predicted, targets, logits = self.forward_step(batch, model)

        return loss.item(), accuracy, labels_predicted, targets, logits

    def training_loop(
        self,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        epoch_num=1,
        scheduler=None,
        grad_clipping_norm=None,
        validate_every=10,
        verbose=True,
        metrics_scorer=None,
    ):
        val_loss = np.inf
        val_accuracy = 0

        epoch_len = len(train_dataloader)

        for epoch in range(epoch_num):
            iter_loader = iter(train_dataloader)
            pbar = self.get_progress_bar(epoch_len)
            model.to(self.device)
            for step_number in range(epoch_len):
                batch = next(iter_loader)
                self._global_steps_num += 1
                train_loss, train_accuracy = self.training_step(
                    model, batch, optimizer, scheduler, grad_clipping_norm
                )

                if step_number % validate_every == 0:
                    batch = next(iter(val_dataloader))
                    val_loss, val_accuracy, _, _, _ = self.test_step(model, batch)
                if verbose:
                    pbar.update(1)
                    pbar.set_description(
                        f'train loss: {round(train_loss, 3)}, ' f'val loss: {round(val_loss, 3)}'
                    )
                if self.tensorboard is not None and step_number % validate_every == 0:
                    self.log_metrics_loss(train_loss, 'train', self._global_steps_num)
                    self.log_metrics_loss(train_loss, 'val', self._global_steps_num)
                    self.log_metrics_accuracy(train_accuracy, 'train', self._global_steps_num)
                    self.log_metrics_accuracy(val_accuracy, 'val', self._global_steps_num)

            # прогоняем модель на тесте/вале и сохраняем если она лучше по accuracy
            predictions, _, targets, _ = self.predict(model, val_dataloader, verbose=verbose)

            accuracy_df = None
            if metrics_scorer is not None:
                accuracy_df = metrics_scorer.get_accuracies_df(targets, predictions)
            accuracy = accuracy_score(targets, predictions)
            logging.info(f'val accuracy after epoch {epoch}: {round(accuracy, 4)}')
            self.do_checkpoint(epoch, accuracy, model, accuracy_df)

    def predict(
        self, model, dataloader, num_iterations=-1, with_all_probas=False, verbose=True,
    ) -> Tuple[List, List, List, List]:
        predictions_all = []
        probas_all = []
        targets_all = []
        all_class_probas = []
        model.to(self.device)
        epoch_len = len(dataloader)
        pbar = self.get_progress_bar(epoch_len)
        for i, batch in enumerate(dataloader):
            _, _, predictions, targets, logits = self.test_step(model, batch)
            soft_probas = torch_softmax(logits, dim=1)
            probas = soft_probas.max(dim=1).values.tolist()
            if with_all_probas:
                all_class_probas.extend(soft_probas.tolist())
            predictions_all.extend(predictions)
            probas_all.extend(probas)
            targets_all.extend(targets)
            if verbose:
                pbar.update(1)
                pbar.set_description('testing')
            if i == num_iterations:
                break

        return predictions_all, probas_all, targets_all, all_class_probas


def predict_img_batch(
    image_paths: Optional[List[str]] = None,
    image_dir: Optional[str] = None,
    batch_size: int = 32,
    model_path: str = '/app/data/models/model_resnet18',
    labels: Optional[List[int]] = None,
    with_all_probas: bool = False,
):
    if (image_paths is None and image_dir is None) or (
        image_paths is not None and image_dir is not None
    ):
        raise ValueError('only one from image_paths or image_dir should be specified')

    if image_dir is not None:
        image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]

    test_df = pd.DataFrame({TorchDataset.img_path_column: image_paths})

    if labels is not None:
        test_df[TorchDataset.label_column] = labels

    dataset = TorchDataset(df=test_df, transformer=get_preprocessor())

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    room_clf, _ = RoomModel.from_pretrained(model_path)

    trainer = TrainerUtils(device='cuda:0')
    predictions_all, probas_all, targets_all, all_class_probas = trainer.predict(
        room_clf, dataloader, with_all_probas=with_all_probas
    )

    return predictions_all, probas_all, targets_all, all_class_probas
