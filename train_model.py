from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import json
import os

# 📂 Paths
train_path = "Dataset/train"
val_path = "Dataset/valid"

# 🔥 Data Augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2
)

val_gen = ImageDataGenerator(rescale=1./255)

# 📊 Load Data
train_data = train_gen.flow_from_directory(
    train_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    val_path,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

# ❗ Check classes
if train_data.num_classes != val_data.num_classes:
    raise ValueError("Train & Validation classes do not match!")

# 💾 Save class names
class_names = list(train_data.class_indices.keys())
os.makedirs("model", exist_ok=True)

with open("model/class_names.json", "w") as f:
    json.dump(class_names, f)

# 🧠 Load Pretrained Model
base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)

# 🔒 Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# 🔥 Custom Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# ⚙️ Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ⏹️ Early Stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# 🚀 Train
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    callbacks=[early_stop]
)

# 🔥 Fine-Tuning (Unfreeze last layers)
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    epochs=5,
    validation_data=val_data
)

# 💾 Save Model
model.save("model/plant_model.keras")

print("✅ High-accuracy model saved successfully!")






















































# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
# import json
# import os

# train_path = "Dataset/train"
# val_path = "Dataset/valid"

# train_gen = ImageDataGenerator(rescale=1./255)
# val_gen = ImageDataGenerator(rescale=1./255)

# train_data = train_gen.flow_from_directory(
#     train_path,
#     target_size=(224,224),
#     batch_size=32,
#     class_mode='categorical'
# )

# val_data = val_gen.flow_from_directory(
#     val_path,
#     target_size=(224,224),
#     batch_size=32,
#     class_mode='categorical'
# )

# # ❗ IMPORTANT CHECK
# if train_data.num_classes != val_data.num_classes:
#     raise ValueError("❌ Train and Validation classes do NOT match!")

# # Save class names
# class_names = list(train_data.class_indices.keys())
# os.makedirs("model", exist_ok=True)

# with open("model/class_names.json", "w") as f:
#     json.dump(class_names, f)

# # Model
# model = Sequential([
#     Input(shape=(224,224,3)),

#     Conv2D(32, (3,3), activation='relu'),
#     MaxPooling2D(),

#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D(),

#     Conv2D(128, (3,3), activation='relu'),
#     MaxPooling2D(),

#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),

#     Dense(train_data.num_classes, activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(train_data, epochs=5, validation_data=val_data)

# model.save("model/plant_model.keras")

# print("✅ Model saved successfully!")