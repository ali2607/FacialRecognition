from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Load the pre-trained emotion detection model
model = load_model('emotion_detection.keras')

# Set the path to the test dataset
test_path = "dataset/test"

# Create the test dataset
test_dataset = image_dataset_from_directory(
    test_path,
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    batch_size=32,
    image_size=(48, 48),
    shuffle=False
)

# Get the class names
class_names = test_dataset.class_names
print("Class Names:", class_names)


# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_dataset)


# Initialize lists to store true and predicted labels
y_true = []
y_pred = []

# Iterate over the test dataset batches
for images, labels in test_dataset:
    # Predict the probabilities for each batch
    predictions = model.predict(images)
    # Convert one-hot encoded labels to integers
    labels = np.argmax(labels.numpy(), axis=1)
    # Get the predicted class indices
    preds = np.argmax(predictions, axis=1)
    # Extend the lists with batch results
    y_true.extend(labels)
    y_pred.extend(preds)

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)


# Compute the confusion matrix
cm = confusion_matrix(y_true, y_pred)
# Plot the confusion matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=class_names,
    yticklabels=class_names,
    cmap='Blues'
)
# Print a detailed classification report
print('Test Loss:', test_loss)
print('Test Accuracy:', test_acc)

report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:\n", report)

plt.title('Confusion Matrix')
plt.xlabel('Predicted Emotion')
plt.ylabel('True Emotion')
plt.show()


