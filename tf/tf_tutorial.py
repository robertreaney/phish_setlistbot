import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

#build tf.keras.Sequential model by stacking layers
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

#convert the logits to probabilities for each class
tf.nn.softmax(predictions).numpy()

#create loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#untrained model should give close to random => 1/10 here
#so initial loss should be -tf.log(1/10) ~= 2.3
loss_fn(y_train[:1], predictions).numpy()

#choose optimizer
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
#train
model.fit(x_train, y_train, epochs=5)

#evaluation
model.evaluate(x_test, y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

preds = probability_model(x_test[:1]).numpy()
preds
















