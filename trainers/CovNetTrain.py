from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np

class CovNetTrainer(BaseTrain):

    def __init__(self, sess, model, data, config, logger):
        super(CovNetTrainer, self).__init__(sess, model, data, config, logger)
    