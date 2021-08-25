# Handwritten Digits Recognition using Artificial Neural Network(ANN)

The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning.

## Importing Libraries
```ruby
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
sns.set()
```
## Loading MNIST Dataset from keras
```ruby
dataset = keras.datasets.mnist.load_data()
```
```ruby
Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
11493376/11490434 [==============================] - 14s 1us/step
11501568/11490434 [==============================] - 14s 1us/step
```
## Exploring The Dataset
```ruby
(x_train,y_train),(x_test,y_test)=dataset
x_train.shape
```ruby
(60000, 28, 28)
```ruby
x_test.shape
```
(10000, 28, 28)
```ruby
plt.matshow(x_train[2])
```
<matplotlib.image.AxesImage at 0x2c52acafd30>
An Image of digit 4
```ruby
y_train[2]
```
4

## Scale the inputs
```ruby
x_train = x_train/255
x_test = x_test/255
```
## Lets' flattened the inputs.( 2 dimention -> 1 dimention)
```ruby
x_train.shape
```
(60000,28,28)
```ruby
x_train_flattened = x_train.reshape(len(x_train),28*28)
x_test_rubyflattened = x_test.reshape(len(x_test),28*28)
```
## Let's create a Neural Network
```ruby
model = kerrubyas.Sequential([
    keras.layers.Dense(10,input_shape=(784,),activation='sigmoid')
])

model.compile(
    optimizer='rubyAdam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train_flattened,y_train, epochs=5)
```
```ruby
Epoch 1/5
1875/1875 [==============================] - 7s 2ms/step - loss: 0.4752 - accuracy: 0.8749
Epoch 2/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.3047 - accuracy: 0.9156
Epoch 3/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.2836 - accuracy: 0.9215
Epoch 4/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.2735 - accuracy: 0.9233
Epoch 5/5
1875/1875 [==============================] - 3s 2ms/step - loss: 0.2670 - accuracy: 0.9256
<keras.callbacks.History at 0x2c52aaa5490>
```
## Evaluate the model
```ruby
model.evaluate(x_test_flattened, y_test)
```
```ruby
313/313 [==============================] - 4s 1ms/step - loss: 0.2681 - accuracy: 0.9248
[0.26814231276512146, 0.9247999787330627]
```
```ruby
92% Accuracy
```
# Check the accuracy by predicting manually
```ruby
pltshow(x_test[998])
```
An Image of digit '8'
```ruby
y_test[998]
```
8
### We have test_input[998] = image of 8 and test_output[998] = 8
**Let's check can our model predict 8 for test_input[998]**
```ruby
y_prediction = model.predict(x_test_flattened)
y_prediction[998]
```
a([2.17765570e-04, 3.28193717e-09, 1.57922506e-04, 8.18024910e-06,
       1.48281455e-02, 2.24798322e-02, 2.45214105e-02, 6.04760647e-03,
       8.48421037e-01, 2.52612501e-01], dtype=float32)
#### It has predicted the probability of each digit (0-9)
```ruby
np.argmax(y_prediction[998])
**argmax finds the maximum and returns the index value**
```
8
## Confusion Matrix
```ruby
tf.math.confusion_matrix(labels=y_test, predictions = y_prediction) y_test are int values but y_predictions are float
```
```ruby
y_predictions_labels = [np.argmax(i) for i in y_prediction]
cm = tf.math.confusion_matrix(labels=y_test, predictions = y_predictions_labels)
cm
```
```ruby
<tf.Tensor: shape=(10, 10), dtype=int32, numpy=
arr58,    0,    2,    2,    0,    5,    9,    2,    2,    0],
       [   0, 1113,    2,    2,    0,    1,    4,    2,   11,    0],
       [   5,    9,  923,   15,    8,    2,   13,   12,   43,    2],
       [   3,    1,   24,  909,    0,   26,    4,   14,   23,    6],
       [   1,    1,    4,    1,  907,    0,   15,    5,   11,   37],
       [   9,    3,    3,   31,   10,  760,   20,   10,   38,    8],
       [   8,    3,    5,    1,    7,    7,  924,    2,    1,    0],
       [   1,    5,   23,    2,    3,    0,    0,  972,    4,   18],
       [   5,    8,    7,   16,    8,   17,   11,   13,  883,    6],
       [  11,    7,    1,   10,   26,    4,    0,   45,    6,  899]])>
```
## For a better view of confusuion matrix
```ruby
plt.figure(figsize = (10,7))
sns.heatmap(cm,annot=True, fmt='d')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
```
![confusion matrix image](https://colab.research.google.com/drive/1EwvuwlVKg9s0BlK2o-WSECfqNZ6bF6tu#scrollTo=2b29cc96&line=1&uniqifier=1)
## Let's add a hidden layer to improve performance
```ruby
model = keras.Sequential([
    keras.layers.Dense(100,input_shape=(784,),activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
])

model.compile(
    optimizer='Adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train_flattened,y_train, epochs=5)
```
```ruby
Epoch 1/5
1875/1875 [==============================] - 6s 3ms/step - loss: 0.2708 - accuracy: 0.9225
Epoch 2/5
1875/1875 [==============================] - 5s 3ms/step - loss: 0.1186 - accuracy: 0.9653
Epoch 3/5
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0835 - accuracy: 0.9749
Epoch 4/5
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0645 - accuracy: 0.9807
Epoch 5/5
1875/1875 [==============================] - 5s 3ms/step - loss: 0.0510 - accuracy: 0.9844
<keras.callbacks.History at 0x2c5468fecd0>
```
## Evaluating the model
```ruby
model.evaluate(x_test_flattened, y_test)
```
```ruby
313/313 [==============================] - 1s 2ms/step - loss: 0.0759 - accuracy: 0.9770
[0.0758744403719902, 0.9769999980926514]
```
```ruby
98% Accuracy
```

# The below code segment is nothing but the same code with keras flatten api. So we dont need to flatten the inputs separately
```ruby
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(
    optimizer='Adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train,y_train, epochs=5)
```
```ruby
Epoch 1/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.2647 - accuracy: 0.9244
Epoch 2/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.1218 - accuracy: 0.9638
Epoch 3/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0862 - accuracy: 0.9737
Epoch 4/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0655 - accuracy: 0.9802
Epoch 5/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0513 - accuracy: 0.9845
<keras.callbacks.History at 0x7f878c62d750>
```
## Evaluating the model
```ruby
model.evaluate(x_test_flattened, y_test)
```
```ruby
98% accuracy
```
# We have achieved 98% accuracy of our ANN model on MNIST dataset.
