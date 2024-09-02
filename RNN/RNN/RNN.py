from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding


model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.summary()

### Num of parameters
# recurrent_weights + input_weights + biases
# or can also be mentioned as = [(num_features + weights)* weights] + biases
# or num_para = units_pre * units + num_bias

# or
# num_units*num_units + num_features*num_units + biases
# This can be written as:
# num_units (num_units + num_features) + biases
# where:
# units_features is the number of features（32 in your settings), 
# num_units is the number of neurons (32 in your settings） in the current layer,
# biases is the number of bias term in the current layer, which is the same as the units.
# (32 + 32) 32 + 32 = 2080


model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
model.summary()

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

max_features = 10000
maxlen = 500
batch_size = 32
print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')
print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation = 'sigmoid'))

# Training logic
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
history = model.fit(input_train, y_train, epochs = 10, batch_size = 128, validation_split = 0.2)


import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


### saving model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")


import pandas as pd
# converting reshaped array to dataframe.
df = pd.DataFrame(input_test)  
# selecting 100 rows randomly
df1 = df.sample(n = 100)
# Saving small amount of data for testing
df1.to_csv('test_rnn.csv', index = False)



# Testing
from tensorflow.keras.models import model_from_json

import pandas as pd
# from keras import model_from_json 

# Opening and store file in a variable

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()


# Use Keras model_from_json to make a loaded model

loaded_model = model_from_json(loaded_model_json)

# Load weights into new model

loaded_model.load_weights("model.h5")
print("Loaded Model from disk")

# Compile and evaluate loaded model

loaded_model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])

test1 = pd.read_csv("test_rnn.csv")

predictions = pd.DataFrame(loaded_model.predict(test1))
predictions
