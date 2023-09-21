import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.matplotlib_fname()

print(tf.__version__)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

r = model.fit([ [2, 2], [10, 10]],
              [ 4,      20],
          validation_data=([ [5, 8], [10, 11]],
                           [ 13,      21],),
          epochs=10)

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.show()
