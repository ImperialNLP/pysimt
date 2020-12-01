import os
import pathlib

from typing import List, Any, Dict

import torch



LANGUAGES = [
    'aa', 'ab', 'ae', 'af', 'ak', 'am', 'an', 'ar', 'as', 'av', 'ay', 'az',
    'ba', 'be', 'bg', 'bh', 'bi', 'bm', 'bn', 'bo', 'br', 'bs', 'ca', 'ce',
    'ch', 'co', 'cr', 'cs', 'cu', 'cv', 'cy', 'da', 'de', 'dv', 'dz', 'ee',
    'el', 'en', 'eo', 'es', 'et', 'eu', 'fa', 'ff', 'fi', 'fj', 'fo', 'fr',
    'fy', 'ga', 'gd', 'gl', 'gn', 'gu', 'gv', 'ha', 'he', 'hi', 'ho', 'hr',
    'ht', 'hu', 'hy', 'hz', 'ia', 'id', 'ie', 'ig', 'ii', 'ik', 'io', 'is',
    'it', 'iu', 'ja', 'jv', 'ka', 'kg', 'ki', 'kj', 'kk', 'kl', 'km', 'kn',
    'ko', 'kr', 'ks', 'ku', 'kv', 'kw', 'ky', 'la', 'lb', 'lg', 'li', 'ln',
    'lo', 'lt', 'lu', 'lv', 'mg', 'mh', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms',
    'mt', 'my', 'na', 'nb', 'nd', 'ne', 'ng', 'nl', 'nn', 'no', 'nr', 'nv',
    'ny', 'oc', 'oj', 'om', 'or', 'os', 'pa', 'pi', 'pl', 'ps', 'pt', 'qu',
    'rm', 'rn', 'ro', 'ru', 'rw', 'sa', 'sc', 'sd', 'se', 'sg', 'si', 'sk',
    'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'ss', 'st', 'su', 'sv', 'sw', 'ta',
    'te', 'tg', 'th', 'ti', 'tk', 'tl', 'tn', 'to', 'tr', 'ts', 'tt', 'tw',
    'ty', 'ug', 'uk', 'ur', 'uz', 've', 'vi', 'vo', 'wa', 'wo', 'xh', 'yi',
    'yo', 'za', 'zh', 'zu']


def fix_seed(seed=None):
    if seed is None:
        seed = time.time()

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return seed


def validate_or_assert(option_name, option_value, valid_options):
    assert option_value in valid_options, \
        f"{option_name!r} should be one of {valid_options!r}"


def get_meteor_jar(ver: str = '1.5'):
    root = pathlib.Path(os.getenv('HOME')) / '.pysimt' / 'meteor-data'
    jar = root / 'meteor-1.5.jar'
    assert jar.exists(), "METEOR not installed, please run 'pysimt-install-extra'"
    return jar


def load_pt_file(fname: str, device: str = 'cpu') -> Dict[str, Any]:
    """Returns saved .(ck)pt file fields."""
    fname = str(pathlib.Path(fname).expanduser())
    data = torch.load(fname, map_location=device)
    if 'history' not in data:
        data['history'] = {}
    return data


def get_language(fname: str) -> str:
    """Heuristic to detect the language from filename components."""
    suffix = pathlib.Path(fname).suffix[1:]
    if suffix not in LANGUAGES:
        logger.info(f"Can not detect language from {fname}, fallback to 'en'")
        suffix = 'en'
    return suffix


def listify(llist):
    """Encapsulate l with list[] if not."""
    return [llist] if not isinstance(llist, list) else llist


def flatten(llist) -> List[Any]:
    return [item for sublist in llist for item in sublist]


def readable_size(n: int) -> str:
    """Return a readable size string."""
    sizes = ['K', 'M', 'G']
    fmt = ''
    size = n
    for i, s in enumerate(sizes):
        nn = n / (1000 ** (i + 1))
        if nn >= 1:
            size = nn
            fmt = sizes[i]
        else:
            break
    return '%.2f%s' % (size, fmt)


def get_module_groups(layer_names: List[str]) -> List[str]:
    groups = set()
    for name in layer_names:
        if '.weight' in name:
            groups.add(name.split('.weight')[0])
        elif '.bias' in name:
            groups.add(name.split('.bias')[0])
    return sorted(list(groups))


def get_n_params(module):
    n_param_learnable = 0
    n_param_frozen = 0

    for param in module.parameters():
        if param.requires_grad:
            n_param_learnable += np.cumprod(param.data.size())[-1]
        else:
            n_param_frozen += np.cumprod(param.data.size())[-1]

    n_param_all = n_param_learnable + n_param_frozen
    return "# parameters: {} ({} learnable)".format(
        readable_size(n_param_all), readable_size(n_param_learnable))
