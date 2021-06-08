import os
import torch
import logging
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from typing import Any, Union, Tuple
from torch.utils.data import DataLoader

from blossom.models import MHAttKWS
from blossom.utils.data_util import *
from blossom.utils.print_util import *
from blossom.datasets import MHAttDataset, _collate_fn

# logging.basicConfig(filename='log.log',level=logging.INFO)

class MHAttKWSLearner():
    def __init__(
        self,
        model: MHAttKWS=None,
        device: str=None
    ) -> None:
        super(MHAttKWSLearner, self).__init__()

        self.model = model
        self.num_classes = self.model.num_classes
        
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def _train(
        self,
        train_dataloader,
        optimizer,
        criterion,
        model_type='binary'
    ):
        self.model.train()

        train_loss = []
        total_sample = 0
        correct = 0
        
        # logging.info(f"[Training]Training start")
        for item in tqdm(train_dataloader):
            x, y = item
            x = x.to(self.device)
            y = y.to(self.device).float()
            
            optimizer.zero_grad()

            output = self.model(x).squeeze(-1) if model_type == 'binary' else self.model(x)

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            total_sample += y.size(0)

            if type == 'multi_class':
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(y.data.view_as(pred)).sum()
            else:
                pred = (torch.sigmoid(output) >= 0.5).long()
                correct += (pred == y).sum().item()
           
        loss = np.mean(train_loss)
        acc = correct / total_sample

        return loss, acc
    
    def _validate(
        self,
        valid_dataloader,
        criterion=None,
        model_type='binary'
    ):
        self.model.eval()

        valid_loss = []
        total_sample =0 
        correct = 0

        with torch.no_grad():
            for item in tqdm(valid_dataloader):
                x, y = item
                x = x.to(self.device)
                y = y.to(self.device).float()
                
                output = self.model(x).squeeze(-1) if model_type == 'binary' else self.model(x)

                if criterion:
                    loss = criterion(output, y)
                    valid_loss.append(loss.item())
                
                total_sample += y.size(0)

                if type == 'multi_class':
                    pred = output.data.max(1, keepdim=True)[1]
                    correct += pred.eq(y.data.view_as(pred)).sum()
                else:
                    pred = (torch.sigmoid(output) >= 0.5).long()
                    correct += (pred == y).sum().item()
         
        loss = np.mean(valid_loss)
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
        print_line("Dataset Info")
        print(f"Length of Training dataset: {len(train_dataset)}")
        print(f"Length of Test dataset: {len(test_dataset)} \n")

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle,
            pin_memory=True, collate_fn=_collate_fn, num_workers=num_workers
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            pin_memory=True, collate_fn=_collate_fn, num_workers=num_workers
        )

        if self.num_classes == 2:
            model_type = 'binary'
        else:
            model_type = 'multi_class'

        self.classes = train_dataset.classes

        criterion = get_build_criterion(model_type=model_type)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

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
            train_loss, train_acc = self._train(train_dataloader, optimizer, criterion, model_type)
            valid_loss, valid_acc = self._validate(test_dataloader, criterion, model_type)

            print_free_style(
                message=f"Epoch {epoch + 1}/{n_epochs}: \n" 
                        f"\t- Train: loss = {train_loss:.4f}; acc = {train_acc:.4f} \n"
                        f"\t- Valid: loss = {valid_loss:.4f}; acc = {valid_acc:.4f} \n"
            )

            if valid_acc > best_acc:
                best_acc = valid_acc
                step = 0
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'classes': self.classes,
                        # 'loss': train_loss,
                    }, 
                    os.path.join(save_path, f"{model_name}.pt")
                )
                print_free_style(f"Save the best model!")
            else:
                step += 1
                if step >= max_steps:
                    break

        print_notice_style(message=f"Path to the saved model: {save_path}/{model_name}.pt")

    def inference(self, input: Union[str, Any]):
        """Inference a given sample. """

        pass

    def load_model(self, model_path):
        """Load the pretrained model

        :param model_path: The path to the pretrained model
        """
        # Check the model file exists
        if not os.path.isfile(model_path):
            raise ValueError(f"The model file `{model_path}` is not exists or broken!")

        checkpoint = torch.load(model_path)
        self.classes = checkpoint['classes']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)


def get_build_criterion(model_type):
    return get_from_registry(model_type, criterion_registry)

criterion_registry = {
    'binary': nn.BCEWithLogitsLoss(),
    'multi_class': nn.CrossEntropyLoss()
}