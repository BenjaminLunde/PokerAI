
import tensorflow as tf
from tensorflow.keras import Sequential, optimizers, layers, Model
from tensorflow.keras.layers import Dense, Input, Flatten

class PokerHandClassifier(Model):
  def __init__(self):
    super(PokerHandClassifier, self).__init__()
    self.layer1 = Dense(10, activation='relu')
    self.layer2 = Dense(10, activation='relu')
    self.outputLayer = Dense(2, activation='softmax')

  def call(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    return self.outputLayer(x)
