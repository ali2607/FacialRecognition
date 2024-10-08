from keras._tf_keras.keras.preprocessing import image_dataset_from_directory
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import RandomZoom, RandomRotation, Rescaling,RandomFlip
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 

# récupérer les paths des images
train_path = "dataset/train"
test_path = "dataset/test"
#préprocesser les images (quelques préprocessing déjà effectuer dans le dataset, mais rescaling nécessaire par exemple) .
#créer un ImageDataGenerator pour les images de train puis lire un flot d'images à partir d'un répertoire .
train_dataset = image_dataset_from_directory(train_path, labels='inferred', label_mode='categorical', color_mode='grayscale', batch_size=32, image_size=(48, 48), shuffle=True, seed=1)
test_dataset = image_dataset_from_directory(test_path, labels='inferred', label_mode='categorical', color_mode='grayscale', batch_size=32, image_size=(48, 48), shuffle=True, seed=1)
print(train_dataset.class_names)
print(test_dataset.class_names) 


data_augmentation = Sequential()
data_augmentation.add(RandomZoom(0.2))
data_augmentation.add(RandomRotation(0.1))
data_augmentation.add(RandomFlip("horizontal"))
data_augmentation.add(Rescaling(1./255))

#récupérer les labels des images
#Créer le réseau de neurones (CNN)
model = Sequential()
model.add(data_augmentation)

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(7, activation='softmax'))

#Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

#récupérer les hyperparamètres (batch_size, epochs, etc)
epochs =10
batch_size = 32

#Entraîner le modèle
history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, batch_size=batch_size)

#Sauvegarder le modèle
model.save('emotion_detection.keras')
