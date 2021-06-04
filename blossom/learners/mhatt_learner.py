import os
import torch
import logging
import numpy as np


from tqdm import tqdm
from typing import Union, Tuple
from torch.utils.data import DataLoader

from blossom.models import MHAttKWS
from blossom.utils.print_util import *
from blossom.datasets import MHAttDataset, _collate_fn

logging.basicConfig(filename='log.log',level=logging.INFO)

class MHAttKWSLearner():
    def __init__(
        self,
        model: MHAttKWS=None,
        device: str=None
    ) -> None:
        super(MHAttKWSLearner, self).__init__()

        self.model = model
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def _train(
        self,
        train_dataloader,
        optimizer,
        criterion,
        epoch
    ):
        self.model.train()

        pbar = tqdm(train_dataloader)
        total_loss = []
        total_sample = 0
        correct = 0
        cur_step = 0
        print_step = 10
        
        logging.info(f"[Training]Training start")
        for batch in pbar:
            optimizer.zero_grad()
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            total_sample += y.size(0)

            pred = logits.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).sum()

            loss.backward()
            optimizer.step()
            cur_step += 1
            
            pbar.set_description("loss: {}\tacc:{}\t".format(sum(total_loss)/cur_step, (correct/total_sample)*100))

            if cur_step % print_step == 0:
                logging.info(
                    f"[Training]Epoch {epoch}\tloss: {total_loss/cur_step}\tacc: {correct/total_sample}"
                )
        
        loss = np.mean(total_loss)
        acc = correct / total_sample

        return loss, acc


    def train(
        self,
        train_dataset: MHAttDataset,
        test_dataset: MHAttDataset,
        batch_size: int=48,
        learning_rate: float=1e-4,
        eps: float=1e-8,
        betas: Tuple[float, float]=(0.9, 0.999),
        max_steps: int=10,
        n_epochs: int=100,
        shuffle: bool=True,
        num_workers: int=8,
        view_model: bool=True,
        save_path: str='./models',
        model_name: str='mhatt_model',
        **kwargs
    ):
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle,
            pin_memory=True, collate_fn=_collate_fn, num_workers=num_workers
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=True, collate_fn=_collate_fn, num_workers=num_workers
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        criterion.to(self.device)
        self.model.to(self.device)

        # View the architecture of the model
        if view_model:
            print_line(text='Model Info')
            print(self.model)

        print(f"Using the device: {self.device}")

        step = 0
        best_acc = 0
        
        print_line(text="Training the model")

        # Check save_path exists
        save_path = os.path.abspath(save_path)
        if not os.path.exists(save_path):
            print(f"Create a folder {save_path}")
            os.mkdir(save_path)

        for epoch in range(n_epochs):
            train_loss, train_acc = self._train(train_dataloader, optimizer, criterion, epoch)

            print_free_style(
                message=f"Epoch {epoch + 1}/{n_epochs}: \n" 
                        f"\t- Train: loss = {train_loss:.4f}; acc = {train_acc:.4f} \n"
            )

            if train_acc > best_acc:
                best_acc = train_acc
                step = 0
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        # 'loss': train_loss,
                    }, 
                    os.path.join(save_path, f"{model_name}.pt")
                )
            else:
                step += 1
                if step >= max_steps:
                    break

        print_notice_style(message=f"Path to the saved model: {save_path}/{model_name}.pt")

    def inference(self, input):
        """Inference a given sample. """

        pass
