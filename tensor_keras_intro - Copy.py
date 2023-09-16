import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
tfds.disable_progress_bar()

# https://www.tensorflow.org/datasets/overview
# https://www.tensorflow.org/datasets/catalog/overview

# print(tfds.list_builders())
# builder = tfds.builder('rock_paper_scissors')
# info = builder.info
# print(info)

##PREPARE ROCK, PAPER, SCISSORS DATA
ds_train = tfds.load(name='rock_paper_scissors:3.0.0', split='train')
ds_test = tfds.load(name='rock_paper_scissors:3.0.0', split='test')








