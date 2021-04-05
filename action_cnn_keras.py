from Dataset import KTH
import keras
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv3D, MaxPooling3D
from keras.models import Sequential

STEPS = 20
BATCH_SIZE = 9
LEARNING_RATE = 5e-4
LEARNING_RATE_UPDATE = 0.9995
KEEP_PROB = 0.9

kth = KTH()
kth.load_from_file()
kth.normalize()

input_shape = (60, 80, 8, 3)
num_classes = 6

kth.train_images = kth.train_images.astype('float32')
kth.test_images = kth.test_images.astype('float32')

print('kth.train_images shape:', kth.train_images.shape)
print(kth.train_images.shape[0], 'kth.train_images samples')
print(kth.test_images.shape[0], 'test samples')

# kth.train_labels = keras.utils.to_categorical(kth.train_labels, num_classes)
# kth.test_labels = keras.utils.to_categorical(kth.test_labels, num_classes)

model = Sequential()
model.add(Conv3D(64, kernel_size=(7, 7, 3), strides=(1, 1, 1), activation='relu', input_shape=input_shape))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same'))

model.add(Conv3D(128, kernel_size=(7, 7, 3), strides=(1, 1, 1), activation='relu', input_shape=input_shape))
model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same', data_format='channels_first'))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(1 - KEEP_PROB))
model.add(Dense(128, activation='relu'))
model.add(Dropout(1 - KEEP_PROB))
model.add(Dense(num_classes, activation='relu'))

model.compile(loss=keras.losses.mean_absolute_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

print(kth.train_images.shape, kth.train_labels.shape)

model.fit(kth.train_images, kth.train_labels,
          batch_size=BATCH_SIZE,
          epochs=STEPS,
          verbose=1,
          validation_data=(kth.test_images, kth.test_labels))

score = model.evaluate(kth.test_images, kth.test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
