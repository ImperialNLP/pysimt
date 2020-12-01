from .metric import Metric
from .multibleu import BLEUScorer
from .sacrebleu import SACREBLEUScorer
from .meteor import METEORScorer
from .rouge import ROUGEScorer
from .simnmt import AVPScorer, AVLScorer, CWMScorer

"""These metrics can be used in early stopping."""

# Generation related metrics
beam_metrics = ["BLEU", "SACREBLEU", "METEOR", "ROUGE"]

metric_info = {
    'BLEU': 'max',
    'SACREBLEU': 'max',
    'METEOR': 'max',
    'ROUGE': 'max',
    'LOSS': 'min',
    'ACC': 'max',
    'RECALL': 'max',
    'PRECISION': 'max',
    'F1': 'max',
    # simultaneous translation
    'AVP': 'min',   # Average proportion (Cho and Esipova, 2016)
    'AVL': 'min',   # Average Lagging (Ma et al., 2019 (STACL))
    'DAL': 'min',   # Differentiable AL (not implemented)
    'CW':  'min',   # Consecutive wait (Gu et al., 2017) [Not Implemented]
}
