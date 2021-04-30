from .greedy import GreedySearch
from .sim_greedy import SimultaneousGreedySearch
from .waitk_greedy import SimultaneousWaitKGreedySearch
from .beam import BeamSearch

def get_translator(_type):
    return {
        'gs': GreedySearch,
        'sgs': SimultaneousGreedySearch,
        'wk': SimultaneousWaitKGreedySearch,
        'bs': BeamSearch,
    }[_type]
