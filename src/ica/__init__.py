"""ICA (Independent Component Analysis) Module"""

from .fastica import FastICA
from .contrast_functions import kurtosis, negentropy, g_logcosh, dg_logcosh

__all__ = ['FastICA', 'kurtosis', 'negentropy', 'g_logcosh', 'dg_logcosh']
