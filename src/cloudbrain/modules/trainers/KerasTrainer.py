import logging
import numpy as np
from keras.utils import np_utils

from cloudbrain.modules.interface import ModuleInterface

_LOGGER = logging.getLogger(__name__)


def _get_class(package, class_name):
    mod = __import__(package, fromlist=[class_name])
    return getattr(mod, class_name)

def _one_hot_encode_object_array(arr):
    """One hot encode a numpy array of objects (e.g. strings)"""

    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

class KerasTrainer(ModuleInterface):
    def __init__(self,
                 subscribers,
                 publishers,
                 classifier_package,
                 classifier_name,
                 classifier_input,
                 classifier_options,
                 classifier_output):

        super(KerasTrainer, self).__init__(subscribers, publishers)
        _LOGGER.debug("Subscribers: %s" % self.subscribers)
        _LOGGER.debug("Publishers: %s" % self.publishers)

        self.classifier_package = classifier_package
        self.classifier_name = classifier_name
        self.classifier_options = classifier_options
        self.classifier_input = classifier_input
        self.classifier_output = classifier_output


    def start(self):

        Classifier = _get_class(self.classifier_package, self.classifier_name)
        model = Classifier(**self.classifier_options)










