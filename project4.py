from sklearn.datasets import load_digits
digits = load_digits()

x = digits.data
y = digits.target

import matplotlib.pyplot as plt
plt.imshow(x[1150].reshape(8,8))
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


model = Sequential()
model.add(Dense(16, input_shape=(64,), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

yc = to_categorical(y)

from sklearn.model_selection import train_test_split
xtr, xts, ytr, yts = train_test_split(x, yc, test_size=0.3)

import numpy as np
train_sizes = (len(xtr) * np.linspace(0.2, 0.9999,4)).astype(int)

acc_scores = []
iw = model.get_weights()

for i in train_sizes:
    xtr2,_,ytr2,_ = train_test_split(xtr, ytr, train_size=i)
    model.set_weights(iw)
    model.fit(xtr2, ytr2, epochs=300, verbose=0)
    acc = model.evaluate(xts, yts)
    acc_scores.append(acc[1])
    print(i)

print(acc_scores)

plt.plot(train_sizes, acc_scores)
plt.show()