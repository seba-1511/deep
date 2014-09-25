from deep.corruption.base import NoCorruption
from deep.corruption.base import BinomialCorruption
from deep.corruption.base import DropoutCorruption
from deep.corruption.base import GaussianCorruption
from deep.corruption.base import SaltPepperCorruption

__all__ = ['NoCorruption',
           'BinomialCorruption',
           'DropoutCorruption',
           'GaussianCorruption',
           'SaltPepperCorruption']