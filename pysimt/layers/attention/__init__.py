from .mlp import MLPAttention
from .dot import DotAttention
from .hierarchical import HierarchicalAttention
from .uniform import UniformAttention
from .scaled_dot import ScaledDotAttention
from .multihead import MultiheadAttention


def get_attention(type_):
    return {
        'mlp': MLPAttention,
        'dot': DotAttention,
        'hier': HierarchicalAttention,
        'uniform': UniformAttention,
        'multihead': MultiheadAttention,
        'scaled_dot': ScaledDotAttention,
    }[type_]
