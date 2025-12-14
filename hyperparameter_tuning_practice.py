from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import tensorflow as tf
import numpy as np
# pip install keras-tuner
# pip install tensorflow
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install scikit-learn
# pip install scipy
# pip install statsmodels
# pip install xgboost

def build_model(hp):
    units = hp.Int(name="units", min_value=16, max_value=64, step=16)
    model = keras.Sequential([
        layers.Dense(units, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    optimizer = hp.Choice(name="optimizer", values=["rmsprop", "adam"])
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    return model


class SimpleMLP(kt.HyperModel):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def build(self, hp):
        units = hp.Int(name="units", min_value=16, max_value=64, step=16)
        model = keras.Sequential([
            layers.Dense(units, activation="relu"),
            layers.Dense(self.num_classes, activation="softmax")
        ])
        optimizer = hp.Choice(name="optimizer", values=["rmsprop", "adam"])
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])
        return model


hypermodel = SimpleMLP(num_classes=10)

tuner = kt.BayesianOptimization(
    build_model,
    objective="val_accuracy",
    max_trials=100,
    executions_per_trial=2,
    directory="mnist_kt_test",
    overwrite=True,
)

tuner.search_space_summary()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28 * 28)).astype("float32") / 255
x_test = x_test.reshape((-1, 28 * 28)).astype("float32") / 255
x_train_full = x_train[:]
y_train_full = y_train[:]
num_val_samples = 10000
x_train, x_val = x_train[:-num_val_samples], x_train[-num_val_samples:]
y_train, y_val = y_train[:-num_val_samples], y_train[-num_val_samples:]
callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
]

tuner.search(
    x_train, y_train,
    batch_size=128,
    epochs=100,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=2,
)

# 顯示所有搜尋結果的摘要
print("\n" + "="*50)
print("搜尋結果摘要:")
print("="*50)
tuner.results_summary()

top_n = 4
best_hps = tuner.get_best_hyperparameters(top_n)

# 顯示前 N 組最佳參數
print("\n" + "="*50)
print(f"前 {top_n} 組最佳參數:")
print("="*50)
for i, hp in enumerate(best_hps):
    print(f"\n第 {i+1} 組:")
    print(f"  units: {hp.get('units')}")
    print(f"  optimizer: {hp.get('optimizer')}")


def get_best_epoch(hp):
    model = build_model(hp)
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=10)
    ]
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=128,
        callbacks=callbacks)
    val_loss_per_epoch = history.history["val_loss"]
    best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
    print(f"Best epoch: {best_epoch}")
    return best_epoch


def get_best_trained_model(hp):
    best_epoch = get_best_epoch(hp)
    model = build_model(hp)
    model.fit(
        x_train_full, y_train_full,
        batch_size=128, epochs=int(best_epoch * 1.2))
    return model


best_models = []
for hp in best_hps:
    model = get_best_trained_model(hp)
    print(f"\n評估模型 - units: {hp.get('units')}, optimizer: {hp.get('optimizer')}")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"  測試準確率: {test_acc:.4f}")
    best_models.append(model)

best_models = tuner.get_best_models(top_n)

# 顯示最終最佳模型
print("\n" + "="*50)
print("最佳模型:")
print("="*50)
best_hp = tuner.get_best_hyperparameters(1)[0]
print(f"  units: {best_hp.get('units')}")
print(f"  optimizer: {best_hp.get('optimizer')}")
print("\n最佳模型測試結果:")
test_loss, test_acc = best_models[0].evaluate(x_test, y_test)
print(f"  測試損失: {test_loss:.4f}")
print(f"  測試準確率: {test_acc:.4f}")
print("="*50)

np_array = np.zeros((2, 2))
tf_tensor = tf.convert_to_tensor(np_array)
print(tf_tensor.dtype)

np_array = np.zeros((2, 2))
tf_tensor = tf.convert_to_tensor(np_array, dtype="float32")
print(tf_tensor.dtype)

keras.mixed_precision.set_global_policy("mixed_float16")
