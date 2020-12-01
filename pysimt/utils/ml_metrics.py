
class Loss:
    """Accumulates and computes correctly training and validation losses."""
    def __init__(self):
        self.reset()

    def reset(self):
        self._loss = 0
        self._denom = 0
        self.batch_loss = 0

    def update(self, loss, n_items):
        # Store last batch loss
        self.batch_loss = loss.item()
        # Add it to cumulative loss
        self._loss += self.batch_loss
        # Normalize batch loss w.r.t n_items
        self.batch_loss /= n_items
        # Accumulate n_items inside the denominator
        self._denom += n_items

    def get(self):
        if self._denom == 0:
            return 0
        return self._loss / self._denom

    @property
    def denom(self):
        return self._denom
