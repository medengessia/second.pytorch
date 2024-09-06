"""
Author : Matthieu Medeng Essia
Mail: matthieu.medeng@altametris.com

"""
from torch.optim import Optimizer

class CustomExponentialLR:
    """
    Implements a customized exponential scheduler to apply the decay factor
    recommended in SECOND's paper every 15 epochs.
    
    Args:
        optimizer (Optimizer): The optimizer to use in the training process.
        decay_factor (float): The exponential decay factor (gamma).
        decay_step (int): The number of epochs after which the decay factor must be applied.
        initial_lr (float): The learning rate.
    """
    def __init__(self, optimizer: Optimizer, decay_factor: float, decay_step: int, initial_lr: float) -> None:
        self.optimizer = optimizer
        self.decay_factor = decay_factor
        self.decay_step = decay_step
        self.initial_lr = initial_lr
        self.epoch = 0

    def step(self) -> None:
        self.epoch += 1
        if self.epoch % self.decay_step == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= self.decay_factor
                print(f"Learning rate updated to {param_group['lr']} at epoch {self.epoch}")

    def state_dict(self):
        # Return the state of the scheduler
        return {
            'decay_factor': self.decay_factor,
            'decay_step': self.decay_step,
            'initial_lr': self.initial_lr,
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict()
        }

    def load_state_dict(self, state_dict):
        # Load the state of the scheduler
        self.decay_factor = state_dict['decay_factor']
        self.decay_step = state_dict['decay_step']
        self.initial_lr = state_dict['initial_lr']
        self.epoch = state_dict['epoch']
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])