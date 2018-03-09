from base.base_train import BaseTrain
import tensorflow as tf
from tqdm import tqdm
import numpy as np

class CovNetTrainer(BaseTrain):

    def __init__(self, sess, model, data, config, logger):
        super(CovNetTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        y_pred = self.model.prediction
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred,
                                                                         labels=tf.one_hot()))
    def train_step(self):
