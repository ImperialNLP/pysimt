from .conditional import ConditionalGRUDecoder
from .tf_decoder import TFDecoder


def get_decoder(type_):
    """Only expose ones with compatible __init__() arguments for now."""
    return {
        'cond': ConditionalGRUDecoder,
        'tf': TFDecoder,
    }[type_]
