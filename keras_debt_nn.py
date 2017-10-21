from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# Input
predictors = np.array([[1, 0, 0],
                       [1, 1, 0],
                       [1, 1, 1],
                       [0, 1, 1],
                       [2, 1, 1],
                       [2, 2, 1],
                       [2, 2, 2],
                       [3, 2, 2],
                       [3, 3, 2],
                       ])
# Output, .T is to change it to transpose the out to a column to match our input
data = np.array([[0, 0, 1, 0,
                  0, 1, 1,
                  0, 1]
                 ]).T
# We change the output to two columns for our two outcomes, 0 and 1
target = to_categorical(data)

# Initialize the model as a Sequential model
model = Sequential()

# We two deep learning layers in our model with 30 and 10 neurons
# We use the the Rectified Linear Unit, relu, as our activation function
model.add(Dense(30, activation='relu', input_shape=(3,)))
model.add(Dense(10, activation='relu'))
# Our output has two nodes for our two outcomes
# We use Softmax as our activation function for output, we want it as 0 or 1
model.add(Dense(2, activation='softmax'))
# Compile our model using adam learning rate optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
# Train the model using our inputs and outputs for 20,000 iterations
model.fit(predictors, target, epochs=20000)
# Get the model prediction for 9 messages sent
test_array = np.array([[3, 3, 3]])
predictions = model.predict(test_array)
print("Model prediction for [3, 3, 3 ] test case")
print(predictions)

# Get predictions for our training data
print("Model predictions for initial training data")
print(model.predict_proba(predictors))