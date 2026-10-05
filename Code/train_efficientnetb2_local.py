"""Train EfficientNetB2 locally on the Figshare brain-tumor dataset."""

from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "Figshare File 1512427"
MODEL_DIR = ROOT / "saved model" / "baseline"
IMAGE_SIZE = (240, 240)
CLASS_NAMES = ("glioma", "meningioma", "pituitary tumor")


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    archives = sorted(DATA_DIR.glob("brainTumorDataPublic_*.zip"))
    if not archives:
        raise FileNotFoundError(f"No dataset archives found in {DATA_DIR}")

    for archive in archives:
        with zipfile.ZipFile(archive) as source:
            for name in source.namelist():
                if not name.endswith(".mat"):
                    continue
                with source.open(name) as mat_file:
                    raw = mat_file.read()
                with h5py.File(__import__("io").BytesIO(raw), "r") as mat:
                    group = mat["cjdata"]
                    image = np.asarray(group["image"], dtype=np.uint8)
                    label = int(np.asarray(group["label"])[0, 0]) - 1
                image = tf.image.resize(image[..., None], IMAGE_SIZE).numpy()
                images.append(np.repeat(image, 3, axis=-1).astype(np.uint8))
                labels.append(label)

    x = np.stack(images)
    y = np.asarray(labels, dtype=np.int32)
    if set(np.unique(y)) != set(range(len(CLASS_NAMES))):
        raise ValueError(f"Unexpected labels: {np.unique(y)}")
    return x, y


def build_model() -> tf.keras.Model:
    backbone = tf.keras.applications.EfficientNetB2(
        weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3)
    )
    backbone.trainable = True
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)
    x = backbone(x, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main(epochs: int) -> None:
    print("TensorFlow:", tf.__version__)
    print("Devices:", tf.config.list_physical_devices())
    start = time.perf_counter()
    x, y = load_dataset()
    print(f"Loaded {len(x)} images: {x.shape}, classes={np.bincount(y)}")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=101, stratify=y
    )
    model = build_model()
    MODEL_DIR.mkdir(exist_ok=True)
    checkpoint = MODEL_DIR / "efficientnetb2_local.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.3, patience=3, min_delta=0.001, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1
        ),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    elapsed = time.perf_counter() - start
    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4%}")
    print(f"Completed {len(history.history['loss'])} epochs in {elapsed / 60:.1f} minutes")
    print(f"Best model: {checkpoint}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    main(parser.parse_args().epochs)
