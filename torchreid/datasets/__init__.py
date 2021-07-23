from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .casiab_video import Casiab
from .casiab_video_sub import Casiab_sub

from .ltcc import Ltcc
from .prcc import Prcc


__imgreid_factory = {
    'ltcc': Ltcc,
    'prcc': Prcc,
}

# gait dataset, for training Gait-Stream
__vidreid_factory = {
    'casiab': Casiab,
    'casiab_processed': Casiab_sub,
}


def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)


def init_vidreid_dataset(name, **kwargs):
    if name not in list(__vidreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__vidreid_factory.keys())))
    return __vidreid_factory[name](**kwargs)