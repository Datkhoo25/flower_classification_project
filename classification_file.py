#import data manipulation packages
import pandas as pd
import numpy as np
import os
import warnings

#import deep learning tools
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import tensorflow
from kerastuner.tuners import RandomSearch

# tensorflow.test.is_gpu_available()
tensorflow.config.list_physical_devices('GPU')

# Set the seed value for experiment reproducibility.
seed = 1842
tensorflow.random.set_seed(seed)
np.random.seed(seed)
# Turn off warnings for cleaner looking notebook
warnings.simplefilter('ignore')

#Load in the data

INPUT_SIZE=224

#define image dataset
#why do we rescale?
image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.2) #shear_range =.25, zoom_range =.2, horizontal_flip = True, rotation_range=20)

#Train & Validation Split
train_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory='data_cleaned/Train',
                                                 shuffle=True,
                                                 target_size=(224, 224),
                                                 subset="training",
                                                 class_mode='categorical')

validation_dataset = image_generator.flow_from_directory(batch_size=32,
                                                 directory='data_cleaned/Train',
                                                 shuffle=True,
                                                 target_size=(224, 224),
                                                 subset="validation",
                                                 class_mode='categorical')

#Organize data for our predictions
image_generator_submission = ImageDataGenerator(rescale=1/255)
fresh_images = image_generator_submission.flow_from_directory(
                                                 directory='data_cleaned/scraped_images',
                                                 shuffle=False,
                                                 target_size=(224, 224),
                                                 class_mode=None)

# #show flowers for the first batch
# batch_1_img = train_dataset[0]
# for i in range(0,32):
#     img = batch_1_img[0][i]
#     lab = batch_1_img[1][i]
#     plt.imshow(img)
#     plt.title(lab)
#     plt.axis('off')
#     plt.show()

# ##Build the first Convolutional Neural Network (CNN)
# model = keras.Sequential([
#     keras.layers.Conv2D(64, 3, activation='relu', input_shape=(224,224,3)),
#     keras.layers.MaxPooling2D(),
#     keras.layers.Dropout(0.5),
#     keras.layers.Flatten(),
#     keras.layers.Dense(64, activation='relu'),
#     keras.layers.Dense(2, activation='softmax')
# ])
#
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# callback = keras.callbacks.EarlyStopping(monitor='val_loss',
#                                          patience=3,
#                                          restore_best_weights=True)
#
# model.fit(train_dataset, epochs=5, validation_data=validation_dataset, callbacks=callback)

##Augment our  Data
data_augmentation = keras.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(224, 224, 3)),
    keras.layers.experimental.preprocessing.RandomRotation(0.1),
    keras.layers.experimental.preprocessing.RandomZoom(0.1),
    keras.layers.RandomContrast(0.1),
    keras.layers.RandomTranslation(0.1, 0.2),
])

# aug_model = keras.Sequential([data_augmentation, model])
#
# aug_model.fit(train_dataset, epochs=5, validation_data=validation_dataset, callbacks=callback)

# loss, accuracy = model.evaluate(validation_dataset)
# print("Loss:", loss)
# print("Accuracy:", accuracy)
#
# loss, accuracy = aug_model.evaluate(validation_dataset)
# print("Loss:", loss)
# print("Accuracy:", accuracy)

#Using Keras Tuner
def build_model(hp):
    i_model = keras.Sequential([data_augmentation])
    i_model.add(keras.layers.AveragePooling2D(pool_size=(4, 4), strides=2))

    for each_layer  in range(hp.Int("Conv Layers", min_value=0, max_value=3)):
        i_model.add(keras.layers.Conv2D(hp.Choice(f"layer_{each_layer}_filters", [8,16,32,64]), 3, activation='relu'))
        i_model.add(keras.layers.MaxPool2D(3, 3))

    i_model.add(keras.layers.Dropout(0.5))
    i_model.add(keras.layers.Flatten())
    i_model.add(keras.layers.Dense(hp.Choice("My Dense layer", [64, 128, 256, 512, 1024]), activation='relu'))
    i_model.add(keras.layers.Dense(2, activation='softmax'))

    i_model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    return i_model


tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=32,
)

tuner.search(train_dataset, validation_data=validation_dataset, epochs=20)
#tuner search ady will split the data into batches and will expect the full data and labels


best_model = tuner.get_best_models()[0]
best_model.summary()
loss, accuracy = best_model.evaluate(validation_dataset)

print("LOSS", loss)
print("ACCURACY", accuracy)

images_file = [f.split('.')[0] for f in os.listdir(os.path.join('data_cleaned/scraped_images/image_files')) if os.path.isfile(os.path.join(os.path.join('data_cleaned/scraped_images/image_files'), f))]
submission_df = pd.DataFrame(images_file, columns=['images'])
submission_df[['la_eterna','other_flower']] = best_model.predict(fresh_images)
submission_df.head()


submission_df.to_csv('submission_file.csv', index=False)
