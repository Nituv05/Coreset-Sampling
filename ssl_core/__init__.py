from ssl_core.base import BaseModel
from ssl_core.simclr import SimCLR
from ssl_core.moco import MoCo
from ssl_core.byol import BYOL
from ssl_core.simsiam import SimSiam
from ssl_core.swav import SwAV
from ssl_core.dino import DINO
from ssl_core.mae import MAE

_method_class_map = {
    'base': BaseModel,
    'simclr': SimCLR,
    'moco': MoCo,
    'byol': BYOL,
    'simsiam': SimSiam,
    'swav': SwAV,
    'dino': DINO,
    'mae': MAE
}


def get_method_class(key):
    if key in _method_class_map:
        return _method_class_map[key]
    else:
        raise ValueError('Invalid method: {}'.format(key))
