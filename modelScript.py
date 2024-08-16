import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.data import AUTOTUNE
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models

# Define the directory path
data_dir = r"skinDisease"

# Set parameters
img_height = 224
img_width = 224
batch_size = 32

# Load the dataset
full_dataset = image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    seed=123,
    label_mode="int",
     interpolation="bilinear"
)

# Define the size for the train, validation, and test datasets
train_size = 0.7
validation_size = 0.15
test_size = 0.15

# Calculate the number of batches
num_batches = tf.data.experimental.cardinality(full_dataset).numpy()
train_batches = int(train_size * num_batches)
validation_batches = int(validation_size * num_batches)

# Split the dataset
train_dataset = full_dataset.take(train_batches)
validation_dataset = full_dataset.skip(train_batches).take(validation_batches)
test_dataset = full_dataset.skip(train_batches + validation_batches)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# Apply data augmentation to the training dataset
augmented_train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
)

# Prefetch for performance
augmented_train_dataset = augmented_train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Example of iterating through the train dataset
for images, labels in augmented_train_dataset:
    print(images.shape)
    print(labels.shape)
    break

# Define the model
model = models.Sequential([
    layers.InputLayer(input_shape=(224, 224, 3)),
    layers.Conv2D(32, (3, 3),padding='same', activation='relu'),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), padding='same',activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), padding='same',activation='relu'),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # layers.Conv2D(256,(3,3),padding='same',activation='relu'),
    # layers.Conv2D(256,(3,3),activation='relu'),
    # layers.MaxPooling2D((2,2)),

    # layers.Conv2D(512,(3,3),padding='same',activation='relu'),
    # layers.Conv2D(512,(3,3),activation='relu'),
    # layers.MaxPooling2D((2,2)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(full_dataset.class_names), activation='softmax')  # Adjust this if you have a different number of classes
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model summary
model.summary()

# Learning Rate Scheduler
# def scheduler(epoch, lr):
#     if epoch < 10:
#         return lr
#     else:
#         return lr * tf.math.exp(-0.1)

# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

# Early Stopping Callback
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model         
history = model.fit(
    augmented_train_dataset,
    validation_data=validation_dataset,
    epochs=15,
    # callbacks=[early_stopping, lr_scheduler]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the model
model.save('skin_mindspring.h5')

# Function to preprocess the input image
def load_and_preprocess_image(img_path, img_height=224, img_width=224):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Function to predict the skin disease
def predict_skin_disease(model, img_path, class_names):
    processed_img = load_and_preprocess_image(img_path)
    predictions = model.predict(processed_img)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]
    return predicted_class_name, confidence

# Define class names (adjust according to your dataset)
class_names = ['Atopic Dermatitis', 'Basal Cell Carcinoma', 'Benign Keratosis-like Lesions', 'Eczema', 'Melanocytic Nevi (NV)', 'Melanoma', 'Psoriasis pictures Lichen Planus and related diseases', 'Seborrheic Keratoses and other Benign Tumors', 'Tinea Ringworm Candidiasis and other Fungal Infections', 'Warts Molluscum and other Viral Infections']

# Predict a skin disease from an image
user_img_path = r"test_img.jpg"
predicted_disease, confidence = predict_skin_disease(model, user_img_path, class_names)
print(f"The model predicts that the image is most likely: {predicted_disease} with a confidence of {confidence:.2f}")
