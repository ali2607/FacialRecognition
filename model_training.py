from keras._tf_keras.keras.preprocessing import image_dataset_from_directory
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import (
    Rescaling,
    RandomFlip,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    RandomTranslation
)

# Set the paths to the training and testing datasets
train_path = "dataset/train"
test_path = "dataset/test"

# Create image datasets from directories with preprocessing
train_dataset,validation_dataset = image_dataset_from_directory(
    train_path,
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=64,
    image_size=(48, 48),
    shuffle=True,
    seed=1,
    validation_split=0.2,
    subset='both' 
)

# Data augmentation and rescaling layer
data_augmentation = Sequential()
data_augmentation.add(RandomTranslation(height_factor=0.1,width_factor=0.1))
data_augmentation.add(RandomFlip("horizontal"))
data_augmentation.add(Rescaling(1.0 / 255))

# Build the convolutional neural network model
model = Sequential()
model.add(data_augmentation)

# First convolutional layer with 32 filters and ReLU activation
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))

model.add(BatchNormalization())

# Second convolutional layer with 64 filters
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Third convolutional layer with 128 filters
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Fourth convolutional layer with 256 filters
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten layer to convert 2D feature maps to 1D feature vectors
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Output layer with 7 units (for 7 emotion classes) and softmax activation
model.add(Dense(7, activation='softmax'))

# Compile the model with Adam optimizer and categorical crossentropy loss
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Print the model summary
print(model.summary())

# Set the hyperparameters
epochs = 200
# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
)

# Save the trained model
model.save('emotion_detection.keras')

