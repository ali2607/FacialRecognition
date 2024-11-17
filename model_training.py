from keras._tf_keras.keras.preprocessing import image_dataset_from_directory
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import (
    Rescaling,
    RandomFlip,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    RandomTranslation
)
import keras_tuner as kt
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau, EarlyStopping

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

# Define the hypermodel
def build_model(hp):
    # Data augmentation and rescaling layer
    data_augmentation = Sequential()
    data_augmentation.add(RandomTranslation(height_factor=0.1,width_factor=0.1))
    data_augmentation.add(RandomFlip("horizontal"))
    data_augmentation.add(Rescaling(1.0 / 255))

    # Build the convolutional neural network model
    model = Sequential()
    model.add(data_augmentation)
    
    # Input layer
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1),padding='same'))
    model.add(BatchNormalization())

    # First convolutional layer with 64 filters
    model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=hp.Float(f'dropout_1', min_value=0.1, max_value=0.5, step=0.1)))


    for i in range(hp.Int('num_conv_layers', 2, 4)):
        model.add(Conv2D(
            filters=hp.Int(f'conv_{i}.1_filters', min_value=64, max_value=512, step=64),
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        ))
        model.add(BatchNormalization())
        model.add(Conv2D(
            filters=hp.Int(f'conv_{i}.2_filters', min_value=64, max_value=512, step=64),
            kernel_size=(3, 3),
            activation='relu',
            padding='same'
        ))
        model.add(BatchNormalization())
        if hp.Choice(f'pooling_{i}', ['max', 'avg']) == 'max':
            model.add(MaxPooling2D(pool_size=(2, 2)))
        else:
            model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))   
    
    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(
        units=hp.Int('dense_units', min_value=128, max_value=512, step=128),
        activation='relu'
    ))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Float('dense_dropout', min_value=0.1, max_value=0.5, step=0.1)))
    
    # Output layer
    model.add(Dense(7, activation='softmax'))
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Instantiate the tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=100,
    executions_per_trial=1,
    directory='tuner_results',
    project_name='emotion_detection_tuning'
)
# Print a summary of the search space
tuner.search_space_summary()

# Perform hyperparameter search
tuner.search(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)
# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

model = tuner.hypermodel.build(best_hps)
# Print the model summary
print(model.summary())
# Set the hyperparameters
epochs = 500
# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    ]
    
)

# Save the trained model
model.save('emotion_detection.keras')

