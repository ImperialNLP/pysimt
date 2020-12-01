import logging

import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_

from .lr_scheduler import NoamScheduler

logger = logging.getLogger('pysimt')


class Optimizer:
    # Class dict to map lowercase identifiers to actual classes
    methods = {
        'adadelta':   torch.optim.Adadelta,
        'adagrad':    torch.optim.Adagrad,
        'adam':       torch.optim.Adam,
        'sgd':        torch.optim.SGD,
        'asgd':       torch.optim.ASGD,
        'rprop':      torch.optim.Rprop,
        'rmsprop':    torch.optim.RMSprop,
    }

    @staticmethod
    def get_params(model):
        """Returns all name, parameter pairs with requires_grad=True."""
        return list(
            filter(lambda p: p[1].requires_grad, model.named_parameters()))

    def __init__(self, name, model, lr=0, momentum=0.0,
                 nesterov=False, weight_decay=0, gclip=0,
                 lr_decay=False, lr_decay_factor=0.1, lr_decay_mode='min',
                 lr_decay_patience=10, lr_decay_min=0.000001, tf_model_dim=512,
                 lr_warmup_steps=4000, adam_betas=(0.9, 0.999)):
        self.name = name
        self.model = model
        self.initial_lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_mode = lr_decay_mode
        self.lr_decay_patience = lr_decay_patience
        self.lr_decay_min = lr_decay_min
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.gclip = gclip
        self.tf_model_dim = tf_model_dim
        self.lr_warmup_steps = lr_warmup_steps
        self.adam_betas = adam_betas

        self.optim_args = {}
        # If an explicit lr given, pass it to torch optimizer
        if self.initial_lr > 0:
            self.optim_args['lr'] = self.initial_lr

        if self.name == 'sgd':
            self.optim_args['momentum'] = self.momentum
            self.optim_args['nesterov'] = self.nesterov

        if self.name == 'adam':
            self.optim_args['betas'] = adam_betas

        # Get all parameters that require grads
        self.named_params = self.get_params(self.model)

        # Filter out names for gradient clipping
        self.params = [param for (name, param) in self.named_params]

        if self.weight_decay > 0:
            weight_group = {
                'params': [p for n, p in self.named_params if 'bias' not in n],
                'weight_decay': self.weight_decay,
            }
            bias_group = {
                'params': [p for n, p in self.named_params if 'bias' in n],
            }
            self.param_groups = [weight_group, bias_group]

        else:
            self.param_groups = [{'params': self.params}]

        # Safety check
        n_params = len(self.params)
        for group in self.param_groups:
            n_params -= len(group['params'])
        assert n_params == 0, "Not all params are passed to the optimizer."

        # Create the actual optimizer
        self.optim = self.methods[self.name](self.param_groups, **self.optim_args)

        # Get final lr that will be used
        self.initial_lr = self.optim.defaults['lr']
        self.cur_lr = self.initial_lr

        # Assign shortcuts
        self.zero_grad = self.optim.zero_grad

        self.step = self._step

        if self.lr_decay == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optim, mode=self.lr_decay_mode,
                factor=self.lr_decay_factor, patience=self.lr_decay_patience,
                min_lr=self.lr_decay_min)
        elif self.lr_decay == 'noam':
            self.scheduler = NoamScheduler(self.optim, tf_model_dim=self.tf_model_dim, learning_rate=lr,
                                           lr_warmup_steps=self.lr_warmup_steps)
        else:
            self.scheduler = None

    def _step(self, closure=None):
        """Gradient clipping aware step()."""
        if self.gclip != 0:
            clip_grad_norm_(self.params, self.gclip)
        if self.lr_decay == 'noam':
            self.scheduler.step()
        else:
            self.optim.step(closure)

    def lr_step(self, metric):
        if self.lr_decay == 'plateau' and self.scheduler is not None:
            self.scheduler.step(metric)
            if self.get_lr() != self.cur_lr:
                self.cur_lr = self.get_lr()
                logger.info('** Learning rate changed -> {}'.format(self.cur_lr))
                # Signal it back
                return True
        return False

    def get_lr(self):
        """Returns current lr for parameters."""
        return self.optim.param_groups[0]['lr']

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def __repr__(self):
        repr_ = "Optimizer => {} (lr: {}, weight_decay: {}, g_clip: {}".format(
            self.name, self.initial_lr, self.weight_decay, self.gclip)
        if self.name == 'sgd':
            repr_ += ', momentum: {}, nesterov: {}'.format(
                self.momentum, self.nesterov)
        if self.name == 'adam':
            repr_ += ', betas: {}'.format(self.adam_betas)
        if self.lr_decay:
            repr_ += ', lr_decay {}: (patience={}, factor={})'.format(
                self.lr_decay, self.lr_decay_patience, self.lr_decay_factor)
        repr_ += ')'
        return repr_
