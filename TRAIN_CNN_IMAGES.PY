
from keras.layers import Dropout,Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop


train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)




training_set = train_datagen.flow_from_directory('images',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


model = Sequential()


model.add(Conv2D(filters=512,kernel_size=(3,3),activation='relu',input_shape=(64,64,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(BatchNormalization())


model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Dropout(0.2))
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))


model.add(BatchNormalization())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))


model.add(BatchNormalization())
model.add(Dense(8,activation='softmax'))


model.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])

print(model.summary())

r = model.fit_generator(
  training_set,
  epochs=5,
  steps_per_epoch=50
)



