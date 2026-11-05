# Import necessary libraries
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
DATASET_DIR = r"C:\Users\HP\Desktop\New ML\Solar-project\DataSolar\Data"
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
SEED = 42
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    image_size = (IMG_HEIGHT,IMG_WIDTH),
    batch_size = BATCH_SIZE,
    seed = SEED,
    color_mode="rgb"
)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split =0.2,
    subset = "validation",
    image_size = (IMG_HEIGHT,IMG_WIDTH),
    batch_size = BATCH_SIZE,
    seed = SEED,
    color_mode="rgb"
)
class_names = train_dataset.class_names
num_classes = len(class_names)
# print(class_names)
# print(num_classes)

# Base MODEL
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Rescaling(1.0/255,input_shape=(IMG_HEIGHT,IMG_WIDTH,3)))
# # Convolutional layer 1
# model.add(tf.keras.layers.Conv2D(32,(3,3),activation="relu"))
# model.add(tf.keras.layers.MaxPool2D((2,2)))
# # Convolutional layer 2
# model.add(tf.keras.layers.Conv2D(64,(3,3),activation="relu"))
# model.add(tf.keras.layers.MaxPool2D((2,2)))
# # Convolutional layer : 3
# model.add(tf.keras.layers.Conv2D(128,(3,3),activation="relu"))
# model.add(tf.keras.layers.MaxPool2D((2,2)))
# # Flattening
# model.add(tf.keras.layers.Flatten())
# # Fully connected layers
# model.add(tf.keras.layers.Dense(128,activation="relu"))
# # Output layer
# model.add(tf.keras.layers.Dense(num_classes,activation="softmax"))
# model.compile(
#     optimizer="adam",
#     loss = "sparse_categorical_crossentropy",
#     metrics = ['accuracy']
# )
# # model.summary()
# EPOCHS = 20
# history = model.fit(
#     train_dataset,
#     validation_data = validation_dataset,
#     epochs = EPOCHS
# )
# # Testing accuracy of 0.9859 and validation accuracy of 0.5706 - means our model is overtraining for 10 epochs
# # Plotting training and validation accuracy
# train_accuracy = history.history['accuracy']
# val_accuracy = history.history['val_accuracy']
# epochs = range(1,EPOCHS+1)
# plt.figure(figsize=(10,5))
# plt.plot(epochs,train_accuracy,label='Training Accuracy',marker = "o")
# plt.plot(epochs,val_accuracy,label = 'Validation Accuracy',marker='x')
# plt.title("Training vs Validation Accuracy")
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
# Trying to change number of epochs to 20 to see if it improves
# After 20 epochs the training accuracy is 0.9972 and validaton accuracy is 0.6215 - An improvement but it is still overfitting

# Trying out early stopping,dropout regularisation and batch normalization
# early_stopping = EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)
# def scheduler(epoch,lr):
#     if epoch == 5:
#         return lr * 0.1
#     return lr
# lr_schedule = LearningRateScheduler(scheduler)

# # MODEL BUILDING 2
# # Base MODEL 2
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Rescaling(1.0/255,input_shape=(IMG_HEIGHT,IMG_WIDTH,3)))
# # Convolutional layer 1
# model.add(tf.keras.layers.Conv2D(32,(3,3),activation="relu"))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.MaxPool2D((2,2)))
# model.add(tf.keras.layers.Dropout(0.3))
# # Convolutional layer 2
# model.add(tf.keras.layers.Conv2D(64,(3,3),activation="relu"))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.MaxPool2D((2,2)))
# model.add(tf.keras.layers.Dropout(0.3))
# # Convolutional layer : 3
# model.add(tf.keras.layers.Conv2D(128,(3,3),activation="relu"))
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.MaxPool2D((2,2)))
# model.add(tf.keras.layers.Dropout(0.3))
# # Flattening
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dropout(0.5))
# # Fully connected layers
# model.add(tf.keras.layers.Dense(128,activation="relu"))
# # Output layer
# model.add(tf.keras.layers.Dense(num_classes,activation="softmax"))
# model.compile(
#     optimizer="adam",
#     loss = "sparse_categorical_crossentropy",
#     metrics = ['accuracy']
# )
# # model.summary()
# EPOCHS = 20
# history = model.fit(
#     train_dataset,
#     validation_data = validation_dataset,
#     epochs = EPOCHS,
#     callbacks=[early_stopping, lr_schedule]
# )
# # Plotting training and validation accuracy
# train_accuracy = history.history['accuracy']
# val_accuracy = history.history['val_accuracy']
# epochs = range(1,EPOCHS+1)
# plt.figure(figsize=(10,5))
# plt.plot(epochs,train_accuracy,label='Training Accuracy',marker = "o")
# plt.plot(epochs,val_accuracy,label = 'Validation Accuracy',marker='x')
# plt.title("Training vs Validation Accuracy")
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# MODEL BUILDING 3

# import os
# import json
# import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt

# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import (
#     classification_report,
#     confusion_matrix,
#     ConfusionMatrixDisplay
# )

# from tensorflow.keras import layers, models, regularizers
# from tensorflow.keras.callbacks import (
#     EarlyStopping,
#     ReduceLROnPlateau,
#     ModelCheckpoint,
#     TerminateOnNaN
# )

# # ==========================================
# # 1. CONFIGURATION
# # ==========================================

# SEED = 42
# IMG_HEIGHT = 224
# IMG_WIDTH = 224
# BATCH_SIZE = 16
# EPOCHS = 40


# tf.keras.utils.set_random_seed(SEED)

# AUTOTUNE = tf.data.AUTOTUNE

# # ==========================================
# # 2. LOAD DATASET
# # ==========================================

# train_dataset = tf.keras.utils.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="training",
#     seed=SEED,
#     image_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     color_mode="rgb",
#     label_mode="int",
#     shuffle=True
# )

# validation_dataset = tf.keras.utils.image_dataset_from_directory(
#     DATASET_DIR,
#     validation_split=0.2,
#     subset="validation",
#     seed=SEED,
#     image_size=(IMG_HEIGHT, IMG_WIDTH),
#     batch_size=BATCH_SIZE,
#     color_mode="rgb",
#     label_mode="int",
#     shuffle=False
# )

# class_names = train_dataset.class_names
# num_classes = len(class_names)

# print("Classes:", class_names)
# print("Number of classes:", num_classes)

# # Optimize data loading without caching
# # the entire dataset in RAM.

# train_dataset = train_dataset.prefetch(AUTOTUNE)
# validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# # ==========================================
# # 3. HANDLE CLASS IMBALANCE
# # ==========================================

# train_labels = np.concatenate([
#     labels.numpy()
#     for images, labels in train_dataset
# ])

# class_weights = compute_class_weight(
#     class_weight="balanced",
#     classes=np.unique(train_labels),
#     y=train_labels
# )

# class_weight_dict = {
#     int(cls): float(weight)
#     for cls, weight in zip(
#         np.unique(train_labels),
#         class_weights
#     )
# }

# print("Class weights:", class_weight_dict)

# # ==========================================
# # 4. DATA AUGMENTATION
# # ==========================================

# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.08),
#     layers.RandomZoom(0.10),
#     layers.RandomContrast(0.10)
# ], name="data_augmentation")

# # ==========================================
# # 5. BUILD CNN MODEL
# # ==========================================

# model = models.Sequential([
    
#     layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

#     # Data augmentation
#     data_augmentation,

#     # Normalize pixel values
#     layers.Rescaling(1.0 / 255),

#     # ------------------------------
#     # CONVOLUTIONAL BLOCK 1
#     # ------------------------------

#     layers.Conv2D(
#         32, (3, 3),
#         padding="same",
#         use_bias=False,
#         kernel_regularizer=regularizers.l2(1e-4)
#     ),
#     layers.BatchNormalization(),
#     layers.Activation("relu"),
#     layers.MaxPooling2D((2, 2)),
#     layers.Dropout(0.15),

#     # ------------------------------
#     # CONVOLUTIONAL BLOCK 2
#     # ------------------------------

#     layers.Conv2D(
#         64, (3, 3),
#         padding="same",
#         use_bias=False,
#         kernel_regularizer=regularizers.l2(1e-4)
#     ),
#     layers.BatchNormalization(),
#     layers.Activation("relu"),
#     layers.MaxPooling2D((2, 2)),
#     layers.Dropout(0.20),

#     # ------------------------------
#     # CONVOLUTIONAL BLOCK 3
#     # ------------------------------

#     layers.Conv2D(
#         128, (3, 3),
#         padding="same",
#         use_bias=False,
#         kernel_regularizer=regularizers.l2(1e-4)
#     ),
#     layers.BatchNormalization(),
#     layers.Activation("relu"),
#     layers.MaxPooling2D((2, 2)),
#     layers.Dropout(0.25),

#     # ------------------------------
#     # CONVOLUTIONAL BLOCK 4
#     # ------------------------------

#     layers.Conv2D(
#         256, (3, 3),
#         padding="same",
#         use_bias=False,
#         kernel_regularizer=regularizers.l2(1e-4)
#     ),
#     layers.BatchNormalization(),
#     layers.Activation("relu"),

#     # Global Average Pooling
#     layers.GlobalAveragePooling2D(),

#     # Fully connected layer
#     layers.Dense(
#         128,
#         activation="relu",
#         kernel_regularizer=regularizers.l2(1e-4)
#     ),

#     layers.Dropout(0.4),

#     # Output layer
#     layers.Dense(
#         num_classes,
#         activation="softmax"
#     )
# ])

# # ==========================================
# # 6. COMPILE MODEL
# # ==========================================

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(
#         learning_rate=0.001
#     ),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # ==========================================
# # 7. CALLBACKS
# # ==========================================

# early_stopping = EarlyStopping(
#     monitor="val_loss",
#     patience=7,
#     restore_best_weights=True,
#     verbose=1
# )

# reduce_lr = ReduceLROnPlateau(
#     monitor="val_loss",
#     factor=0.5,
#     patience=2,
#     min_lr=1e-6,
#     verbose=1
# )

# checkpoint = ModelCheckpoint(
#     "best_solar_model.keras",
#     monitor="val_loss",
#     save_best_only=True,
#     verbose=1
# )

# callbacks = [
#     early_stopping,
#     reduce_lr,
#     checkpoint,
#     TerminateOnNaN()
# ]

# # ==========================================
# # 8. TRAIN MODEL
# # ==========================================

# history = model.fit(
#     train_dataset,
#     validation_data=validation_dataset,
#     epochs=EPOCHS,
#     callbacks=callbacks,
#     class_weight=class_weight_dict
# )

# # ==========================================
# # 9. SAVE MODEL AND CLASS NAMES
# # ==========================================

# model.save("solar_classifier_final.keras")

# with open("class_names.json", "w") as file:
#     json.dump(class_names, file)

# # ==========================================
# # 10. EVALUATE MODEL
# # ==========================================

# val_loss, val_accuracy = model.evaluate(
#     validation_dataset,
#     verbose=1
# )

# print("\nValidation Loss:", val_loss)
# print("Validation Accuracy:", val_accuracy)

# # ==========================================
# # 11. PLOT TRAINING ACCURACY
# # ==========================================

# epochs_range = range(1, len(history.history["accuracy"]) + 1)

# plt.figure(figsize=(10, 5))

# plt.plot(
#     epochs_range,
#     history.history["accuracy"],
#     label="Training Accuracy"
# )

# plt.plot(
#     epochs_range,
#     history.history["val_accuracy"],
#     label="Validation Accuracy"
# )

# plt.title("Training vs Validation Accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid()
# plt.show()

# # ==========================================
# # 12. PLOT TRAINING LOSS
# # ==========================================

# plt.figure(figsize=(10, 5))

# plt.plot(
#     epochs_range,
#     history.history["loss"],
#     label="Training Loss"
# )

# plt.plot(
#     epochs_range,
#     history.history["val_loss"],
#     label="Validation Loss"
# )

# plt.title("Training vs Validation Loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid()
# plt.show()

# # ==========================================
# # 13. CLASSIFICATION REPORT
# # ==========================================

# y_true = np.concatenate([
#     labels.numpy()
#     for images, labels in validation_dataset
# ])

# predictions = model.predict(validation_dataset)

# y_pred = np.argmax(predictions, axis=1)

# print("\nClassification Report:\n")

# print(
#     classification_report(
#         y_true,
#         y_pred,
#         target_names=class_names,
#         zero_division=0
#     )
# )

# # ==========================================
# # 14. CONFUSION MATRIX
# # ==========================================

# cm = confusion_matrix(y_true, y_pred)

# disp = ConfusionMatrixDisplay(
#     confusion_matrix=cm,
#     display_labels=class_names
# )

# fig, ax = plt.subplots(figsize=(10, 8))

# disp.plot(
#     cmap="Blues",
#     xticks_rotation=45,
#     ax=ax
# )

# plt.title("Solar Classification Confusion Matrix")
# plt.tight_layout()
# plt.show()

# This didnt produce the accuracy we want..we go to transfer learning
# TRANSFER LEARNING - MobileNet v2
# base_model = tf.keras.applications.MobileNetV2(
#     input_shape = (224,224,3),
#     include_top = False,
#     weights = 'imagenet'
# )
# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip('horizontal'),
#     tf.keras.layers.RandomRotation(0.15),
#     tf.keras.layers.RandomZoom(0.15)
# ])
# base_model.trainable = False
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Input(shape=(224,224,3)))
# model.add(data_augmentation)
# model.add(tf.keras.layers.Rescaling(1.0/255))
# model.add(base_model)
# model.add(tf.keras.layers.GlobalAveragePooling2D())
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dense(num_classes,activation='softmax'))
# # Compiling the model
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics = ['accuracy']
# )
# model.summary()
# history = model.fit(
#     train_dataset,
#     validation_data = validation_dataset,
#     epochs=20
# )
# model.save('mobilev2_2.h5')
# We see a validation accuracy of 0.7288 as the highest - still not good
# TRANSFER LEARNING : 2
# EFFICIENTNETB0
# from tensorflow.keras.applications.efficientnet import EfficientNetB0
# from tensorflow.keras.applications.efficientnet import preprocess_input
import os 
class_counts = {}
total_images = 0
for class_name in class_names:
    class_path = os.path.join(DATASET_DIR,class_name)
    count = len(os.listdir(class_path))
    class_counts[class_name] = count
    total_images  += count
class_weights = {}
for index,class_name in enumerate(class_names):
    class_weights[index] = total_images / (num_classes * class_counts[class_name])
# # print("Class weights:",class_weights)
# data_augmentation = tf.keras.Sequential()
# data_augmentation.add(tf.keras.layers.RandomFlip("horizontal"))
# data_augmentation.add(tf.keras.layers.RandomRotation((0.1)))
# data_augmentation.add(tf.keras.layers.RandomZoom((0.1)))
# train_images = []
# train_labels = []
# for images,labels in train_dataset:
#     images = preprocess_input(images)
#     train_images.append(images)
#     train_labels.append(labels)
# train_images = tf.concat(train_images,axis=0)
# train_labels = tf.concat(train_labels,axis=0)
# validation_images = []
# validation_labels = []
# for images,labels in validation_dataset:
#     images = preprocess_input(images)
#     validation_images.append(images)
#     validation_labels.append(labels)
# validation_images = tf.concat(validation_images,axis=0)
# validation_labels = tf.concat(validation_labels,axis=0)

# base_model = tf.keras.applications.EfficientNetB0(
#     input_shape=(224,224,3),
#     include_top=False,
#     weights='imagenet'
# )
# base_model.trainable = False
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Input(shape=(224,224,3)))
# model.add(data_augmentation)
# model.add(tf.keras.layers.Rescaling(1.0/255))
# model.add(base_model)
# model.add(tf.keras.layers.GlobalAveragePooling2D())
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dense(num_classes,activation='softmax'))
# # Compiling the model
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics = ['accuracy']
# )
# model.summary()
# history = model.fit(
#     train_dataset,
#     validation_data = validation_dataset,
#     epochs=20
# )

import keras_tuner as kt
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Reuse your train_dataset and validation_dataset (with preprocess_input already applied)
# Make sure they are defined as in your original code

def build_model(hp):
    base_model = EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = Sequential([
        RandomFlip("horizontal"),
        RandomRotation(hp.Float("rotation_factor", min_value=0.05, max_value=0.3, step=0.05)),
        RandomZoom(hp.Float("zoom_factor", min_value=0.05, max_value=0.3, step=0.05)),
        base_model,
        GlobalAveragePooling2D(),
        Dropout(hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.1)),
        Dense(
            hp.Int("dense_units", min_value=64, max_value=512, step=64),
            activation='relu'
        ),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Setup Random Search Tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,          # Same as n_iter before
    executions_per_trial=1,
    directory='kt_dir',
    project_name='efficientnet_tune',
    overwrite=False
)

# Optional: Early stopping
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Run the search
tuner.search(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20,
    callbacks=[early_stopping],
    class_weight=class_weights  # Handles imbalance
)

# Results
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("Best hyperparameters:")
print(best_hps.values)

best_model = tuner.get_best_models(num_models=1)[0]

# Evaluate on validation
val_loss, val_acc = best_model.evaluate(validation_dataset)
print(f"Best validation accuracy: {val_acc:.4f}")
best_model.save(r'C:\Users\HP\Desktop\New ML\Solar-project\trained_effnet_finetune.h5')
# We got the best validation accuracy as 0.825 - thats nice