import tensorflow as tf

# Build the neural network
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(256, input_shape=(784,)),
  tf.keras.layers.Activation('relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(128),
  tf.keras.layers.Activation('relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10),
  tf.keras.layers.Activation('softmax'),
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate the model
model.evaluate(x_test, y_test)
