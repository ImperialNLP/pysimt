class NoamScheduler:
    """NoamScheduler implementation from the `Attention is all you need!` paper."""
    def __init__(self, optimizer, tf_model_dim, learning_rate, lr_warmup_steps=4000):
        """
        Creates a NoamScheduler, implementing the formula from the Attention is all you need! paper.
        :param optimizer: The optimizer.
        :param tf_model_dim: The model dimensions.
        :param learning_rate: The learning rate.
        :param lr_warmup_steps: The warmup steps.
        """
        assert tf_model_dim is not None, 'tf_model_dim must be set to the model dimensions noam decay'
        assert lr_warmup_steps > 0, 'lr_warmup_steps must be greater than 0 for noam decay'
        self.optimizer = optimizer
        self._num_steps = 0
        self.lr_warmup_steps = lr_warmup_steps
        self.tf_model_dim = tf_model_dim
        self._learning_rate = learning_rate

    def step(self):
        """
        Reduces the learning rate according to the formula in Attention is all you need! and performs an optimizer step.
        """
        self._num_steps += 1
        current_learning_rate = self.get_decay() * self._learning_rate
        for parameter in self.optimizer.param_groups:
            parameter['lr'] = current_learning_rate
        self.optimizer.step()

    def get_decay(self):
        return self.tf_model_dim ** (-0.5) * min(self._num_steps ** (-0.5),
                                                 self._num_steps * self.lr_warmup_steps ** (-1.5))
