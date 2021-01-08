import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import Sequential, optimizers, layers, Model
from tensorflow.keras.layers import Dense, Input, Flatten

import PokerHandClassifier as PHC


COLUMN_NAMES = [
    'Suit1', 'Rank1',
    'Suit2', 'Rank2',
    'Suit3', 'Rank3',
    'Suit4', 'Rank4',
    'Suit5', 'Rank5',
    'Hand'
]

#Getting the traning data
current_path = os.path.dirname(__file__) # Where your .py file is located
traning_data = pd.read_csv(os.path.join(current_path,"poker-hand-testing.csv"), names=COLUMN_NAMES, header=0)
traning_data.head()

#Fixing data types
traning_data['Hand'] = traning_data['Hand'].astype("category")

#Normalizing data

traning_data["Suit1"] = traning_data["Suit1"]/4
traning_data["Suit2"] = traning_data["Suit2"]/4
traning_data["Suit3"] = traning_data["Suit3"]/4
traning_data["Suit4"] = traning_data["Suit4"]/4
traning_data["Suit5"] = traning_data["Suit5"]/4

traning_data["Rank1"] = traning_data["Rank1"]/13
traning_data["Rank2"] = traning_data["Rank2"]/13
traning_data["Rank3"] = traning_data["Rank3"]/13
traning_data["Rank4"] = traning_data["Rank4"]/13
traning_data["Rank5"] = traning_data["Rank5"]/13

#Getting correlations
"""
corrMatt = traning_data[['Suit1', 'Rank1','Suit2', 'Rank2','Suit3', 'Rank3','Suit4', 'Rank4','Suit5', 'Rank5','Hand']].corr()
mask = np.array(corrMatt)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corrMatt, mask=mask,vmax=.8, square=True,annot=True) """

#Spliting the data
output_data = traning_data["Hand"]
input_data = traning_data.drop("Hand",axis=1)
x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.3, random_state=42)


#X_train_norm = tf.keras.utils.normalize(X_train.values, axis=-1, order=2)
#X_test_norm = tf.keras.utils.normalize(X_test.values, axis=-1, order=2)


#Creating the nureal network
model = PHC.PokerHandClassifier()

model.compile(optimizer= tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

model.fit(x_train, y_train, epochs=2, batch_size=4240)

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

scores = model.evaluate(x_test, y_test)

print("\nAccuracy: %.2f%%" % (scores[1]*100))


prediction = model.predict(x_test)
prediction1 = pd.DataFrame({'Loss':prediction[:,0],'Win':prediction[:,1]})
prediction1.round(decimals=4).head()

real = pd.DataFrame(y_test)

print(prediction1)
print(real)


# Save checkpoints
model.save_weights('./checkpoints/my_checkpoint')