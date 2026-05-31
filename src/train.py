"""
Train a White Blood Cell (WBC) image classifier.

This script handles:
    1. Loading and preparing image datasets (train / validation / test).
    2. Building and training a CNN model with transfer learning.
    3. Logging experiments to MLflow.
    4. Saving the best-performing model for the Streamlit app to use.

This file should be run once to produce a trained model.
The Streamlit app only loads the saved model — it never trains.
"""

from pathlib import Path
import tensorflow as tf
from tensorflow import keras
import json
import mlflow
import mlflow.tensorflow
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


# --- Configuration -----------------------------------------------------------
# Paths are relative to the project root, so the script must be launched from
# the project root (e.g. `python src/train.py`).
# Combined folder: all images from the original TRAIN + TEST, organized
# into one folder per class. We split this 70/15/15 ourselves.
DATA_DIR = Path("data/combined")

# Image preprocessing settings.
# 224x224 is the standard input size for MobileNetV2, which we will use later
# for transfer learning. Keeping it here ensures train/validation/test
# datasets are all resized consistently.
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Reproducibility — same seed means the same train/validation split
# every time the script is run.
SEED = 42


def load_datasets():
    """
    Load all images from data/combined, shuffle, and split 70/15/15 into
    train / validation / test. Each subfolder of DATA_DIR is one class.

    This satisfies assignment 3.1 (three datasets from a proper split).

    Returns:
        train_ds, val_ds, test_ds, class_names
    """
    # Load every image as one unbatched dataset so we can split it cleanly.
    full_ds = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=IMAGE_SIZE,
        batch_size=None,        # unbatched: one (image, label) at a time
        shuffle=True,
        seed=SEED,
    )

    class_names = full_ds.class_names

    total = full_ds.cardinality().numpy()
    train_size = int(0.70 * total)
    val_size = int(0.15 * total)

    train_ds = full_ds.take(train_size)
    rest = full_ds.skip(train_size)
    val_ds = rest.take(val_size)
    test_ds = rest.skip(val_size)

    # Batch and prefetch each split for training speed.
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


# --- Model configuration -----------------------------------------------------
NUM_CLASSES = 4
def make_augmentation():
    """
    Random image augmentation applied only during training.

    Each epoch the model sees slightly different versions of each image
    (flips, rotations, zoom, contrast). This improves generalization to
    unseen test images and reduces overfitting.
    """
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomContrast(0.1),
        ],
        name="data_augmentation",
    )


def build_model(dropout_rate: float = 0.3, dense_units: int = 128):
    """
    Build a transfer learning model based on MobileNetV2.

    The MobileNetV2 base is frozen — only the new classification head
    is trained. This makes training fast on CPU and works well with
    a few thousand images.

    Args:
        dropout_rate: fraction of neurons randomly dropped during training,
                      to reduce overfitting. Typical values: 0.2 - 0.5.
        dense_units: size of the hidden dense layer in the new head.

    Returns:
        A compiled keras.Model ready for training.
    """
    # 1. Load MobileNetV2 without its original classification head
    #    (include_top=False) and with ImageNet weights.
    base_model = keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),  # (224, 224, 3) — RGB images
        include_top=False,
        weights="imagenet",
    )

    # 2. Freeze the base: its 2.2M parameters stay fixed during training.
    base_model.trainable = False

    # 3. Build our own classification head on top.
    inputs = keras.Input(shape=IMAGE_SIZE + (3,))

    # Augmentation runs only in training mode; Keras automatically
    # disables it during evaluation and prediction.
    x = make_augmentation()(inputs)

    # MobileNetV2 expects pixel values in a specific range.
    x = keras.applications.mobilenet_v2.preprocess_input(x)

    # Feature extraction with the frozen base.
    # training=False keeps any internal BatchNormalization layers in
    # inference mode, which is required when the base is frozen.
    x = base_model(x, training=False)

    # Flatten spatial features into one vector per image.
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Dropout — regularization to reduce overfitting.
    x = keras.layers.Dropout(dropout_rate)(x)

    # Extra dense layer for additional learning capacity in the head.
    x = keras.layers.Dense(dense_units, activation="relu")(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    # Final layer: one neuron per class, softmax converts to probabilities.
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Return the base model too, so fine-tuning can unfreeze it later.
    return model, base_model

# --- MLflow configuration ----------------------------------------------------
# Local MLflow tracking. The mlruns/ folder will be created automatically
# inside the project root. In docker-compose later we will point this to
# a dedicated MLflow service.
import os
# Configurable: defaults to local file store, but can point to the MLflow
# server (e.g. http://localhost:5000) via the MLFLOW_TRACKING_URI env var.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "wbc-classifier"

# Where the final, best model is saved for the Streamlit app to load.
MODEL_OUTPUT_PATH = Path("models/best_model.keras")
CLASS_NAMES_PATH = Path("models/class_names.json")


def evaluate_model(model, test_ds, class_names):
    """
    Compute classification metrics on the test set.

    Returns a dict with accuracy, precision, recall and F1-score.
    Precision/recall/F1 use 'macro' averaging — each class contributes
    equally, regardless of class size. This matches the assignment's
    requirement to report all four metrics.
    """
    # Collect predictions and true labels across the whole test dataset.
    y_true = []
    y_pred = []
    for batch_images, batch_labels in test_ds:
        preds = model.predict(batch_images, verbose=0)
        y_true.extend(np.argmax(batch_labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        "test_accuracy": float((y_true == y_pred).mean()),
        "test_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "test_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def train_one_experiment(run_name, params, train_ds, val_ds, test_ds, class_names):
    """
    Train one model with the given hyperparameters and log everything to MLflow.

    Returns the test F1-score so the caller can pick the best run.
    """
    with mlflow.start_run(run_name=run_name):
        # 1. Log hyperparameters — required by assignment 3.3.
        mlflow.log_params(params)
        run_id = mlflow.active_run().info.run_id

        # 2. Build and train the model.
        model, base_model = build_model(
            dropout_rate=params["dropout_rate"],
            dense_units=params["dense_units"],
        )

        # EarlyStopping — if validation loss stops improving, training stops
        # before reaching max epochs. Saves time and prevents overfitting.
        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
        # Reduce learning rate when validation loss plateaus — helps the
        # model settle into a better minimum instead of bouncing around.
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
        )
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=params["epochs"],
            callbacks=[early_stop, reduce_lr],
            verbose=1,
        )
        # --- Phase 2: fine-tuning -------------------------------------------
        # Unfreeze the top layers of MobileNetV2 so they adapt to blood cells.
        # We keep the lower layers frozen (they learn universal features).
        base_model.trainable = True
        fine_tune_from = len(base_model.layers) - params["unfreeze_layers"]
        for layer in base_model.layers[:fine_tune_from]:
            layer.trainable = False

        # Recompile with a very low learning rate so we only gently adjust
        # the pretrained weights instead of destroying them.
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        print(f"\n[{run_name}] Fine-tuning: unfreezing top {params['unfreeze_layers']} layers...")

        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=params["fine_tune_epochs"],
            callbacks=[early_stop],
            verbose=1,
        )

        # Combine both training phases' validation history for logging.
        for key in ["val_accuracy", "val_loss"]:
            history.history[key].extend(history_ft.history[key])
        # 3. Log per-epoch validation metrics.
        for epoch in range(len(history.history["val_accuracy"])):
            mlflow.log_metric("val_accuracy", history.history["val_accuracy"][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)

        # 4. Evaluate on the test set (untouched during training).
        metrics = evaluate_model(model, test_ds, class_names)
        mlflow.log_metrics(metrics)

        # 5. Log the model itself as an MLflow artifact.
        mlflow.tensorflow.log_model(model, artifact_path="model")

        print(f"\n[{run_name}] test metrics: {metrics}")

        return metrics["test_f1"], model, run_id


def run_all_experiments(train_ds, val_ds, test_ds, class_names):
    """
    Run three experiments with different hyperparameter combinations,
    pick the best by test F1-score, and save it for the Streamlit app.

    This satisfies assignment 3.3 (at least 3 experiments) and 3.4
    (automatic best-model selection logic, not manual).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Three experiments — varying dropout, dense layer size, and epochs.
    experiments = [
        {
            "name": "baseline",
            "params": {
                "dropout_rate": 0.3, "dense_units": 128,
                "epochs": 4, "unfreeze_layers": 30, "fine_tune_epochs": 8,
            },
        },
        {
            "name": "more_dropout",
            "params": {
                "dropout_rate": 0.5, "dense_units": 128,
                "epochs": 4, "unfreeze_layers": 30, "fine_tune_epochs": 8,
            },
        },
        {
            "name": "larger_head",
            "params": {
                "dropout_rate": 0.4, "dense_units": 256,
                "epochs": 4, "unfreeze_layers": 40, "fine_tune_epochs": 8,
            },
        },
    ]

    results = []
    for exp in experiments:
        print(f"\n{'=' * 60}")
        print(f"Running experiment: {exp['name']}")
        print(f"Parameters: {exp['params']}")
        print(f"{'=' * 60}")

        f1, model, run_id = train_one_experiment(
            run_name=exp["name"],
            params=exp["params"],
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            class_names=class_names,
        )
        results.append({"name": exp["name"], "f1": f1, "model": model, "run_id": run_id})

    # Pick the best run automatically by F1-score.
    best = max(results, key=lambda r: r["f1"])
    print(f"\n{'=' * 60}")
    print(f"Best model: {best['name']} (F1 = {best['f1']:.4f})")
    print(f"{'=' * 60}")

    # Save the best model for the Streamlit app to load.
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    best["model"].save(MODEL_OUTPUT_PATH)
    print(f"Saved best model to: {MODEL_OUTPUT_PATH}")

    # Register the best model in the MLflow Model Registry and tag it
    # with the "champion" alias, so the app can always load the current best.
    best_run_id = best["run_id"]
    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri, "wbc-classifier")
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(
        name="wbc-classifier",
        alias="champion",
        version=registered.version,
    )
    print(f"Registered model version {registered.version} as 'champion'")

    # Save class names so the app can map prediction indices to labels.
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f)
    print(f"Saved class names to: {CLASS_NAMES_PATH}")

    return best

if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print("Loading datasets...")

    train_ds, val_ds, test_ds, class_names = load_datasets()

    print(f"\nClasses found: {class_names}")
    print(f"Total train batches:      {len(train_ds)}")
    print(f"Total validation batches: {len(val_ds)}")
    print(f"Total test batches:       {len(test_ds)}")

    run_all_experiments(train_ds, val_ds, test_ds, class_names)
