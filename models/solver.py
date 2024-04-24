import os
import torch
import numpy as np

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from utils.utils import instantiate_from_config, cycle

class Trainer(object):
    def __init__(self, config, model, dataloader):
        super().__init__()
        self.model = model
        self.device = self.model.betas.device
        self.train_num_steps = config['solver']['max_epochs'] # ?
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader)
        self.step = 0
        self.milestone = 0

        self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        self.log_frequency = 100

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
        }
        path = f"{self.results_folder}/checkpoint-{milestone}.pt"
        torch.save(data, path)

    def load(self, milestone):
        path = f"{self.results_folder}/checkpoint-{milestone}.pt"
        data = torch.load(path, map_location=self.device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.milestone = milestone

    def train(self):
        device = self.device
        step = 0
        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                # total_loss = 0.
                # for _ in range(self.gradient_accumulate_every):
                #     data = next(self.dl).to(device)
                #     loss = self.model(data, target=data)
                #     loss = loss / self.gradient_accumulate_every
                #     loss.backward()
                #     total_loss += loss.item()
                    
                data = next(self.dl).to(device)
                loss = self.model(data, target=data)
                loss.backward()
                loss = loss.item()
                
                pbar.set_description(f'loss: {loss:.6f}')
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(loss)
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()

                # save milestone
                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        self.save(self.milestone)
                pbar.update(1)
        print('training complete')

    def sample(self, num, size_every, shape):
        # shape = [24, 6] 
        # num = 10000
        # size_every = 2001
        
        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        for _ in range(num_cycle):
            sample = self.ema.ema_model.generate_mts(batch_size=size_every)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()

        return samples
