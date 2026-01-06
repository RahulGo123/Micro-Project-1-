import tensorflow as tf
import numpy as np
import os
import sys

# Ensure imports work if run directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from database.db_manager import log_keras_history

# ---------------------------------------------------------
# FIX 1: Load the Correct Dataset (Fashion MNIST)
# ---------------------------------------------------------
(X_train_full, y_train_full), (X_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

# Normalization
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------
# FIX 2: Correct Architecture (LeakyReLU as a Layer)
# ---------------------------------------------------------
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=[28, 28]),
        tf.keras.layers.BatchNormalization(),
        # Hidden Layer 1: Dense -> BN -> Activation
        tf.keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),  # Activation is a separate layer now!
        # Hidden Layer 2: Dense -> BN -> Activation
        tf.keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        # Output Layer
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

# Optimizer & Scheduling
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.01)

model.compile(
    loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
)

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=0.0001
)

# Callbacks
early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=20)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "my_mnist_model.keras", save_best_only=True
)

# Training
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_data=(X_valid, y_valid),
    callbacks=[checkpoint_cb, early_stopping_cb],
)

# Evaluation
model = tf.keras.models.load_model("my_mnist_model.keras")
print(f"Accuracy on test set {model.evaluate(X_test, y_test)}")

# Logging
log_keras_history(
    "Day5_Advanced_Stable",
    {"opt": "Nadam", "LR_Sched": "ReduceOnPlateau", "Init": "He", "BN": "True"},
    history,
)
