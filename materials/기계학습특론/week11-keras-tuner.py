import keras
from keras import layers
import numpy as np
import os
import keras_tuner

# 데이터 로드
def load_data():
    # 데이터 파일 경로
    train_images_path = "./data/train-images.idx3-ubyte"
    train_labels_path = "./data/train-labels.idx1-ubyte"
    test_images_path = "./data/t10k-images.idx3-ubyte"
    test_labels_path = "./data/t10k-labels.idx1-ubyte"

    # 데이터를 로드하는 코드
    def load_mnist_images(file_path):
        with open(file_path, "rb") as f:
            # skip magic number and number of images
            f.read(16)
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(-1, 28, 28)
        return data

    # MNIST 라벨 데이터를 로드
    def load_mnist_labels(file_path):
        with open(file_path, "rb") as f:
            f.read(8)
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    # 훈련, 테스트 데이터 로드
    x_train = load_mnist_images(train_images_path)
    y_train = load_mnist_labels(train_labels_path)
    x_test = load_mnist_images(test_images_path)
    y_test = load_mnist_labels(test_labels_path)

    # 데이터 분할
    x_train, x_val = x_train[:-10000], x_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # 데이터 전처리
    x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
    x_val = np.expand_dims(x_val, -1).astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

    # 라벨 데이터
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

# 모델 빌드 함수
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            units=hp.Int("units", min_value=32, max_value=512, step=32), 
            activation=hp.Choice("activation", ["relu", "tanh"]),
        )
    )
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(10, activation="softmax"))  # 출력 레이어

    # 학습률을 튜닝하여 Adam 옵티마이저 설정
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# 하이퍼파라미터 튜너 설정
tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)

# 데이터 로드
(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data()

# 튜닝 시작
tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))

# 모델들을 가져오기
models = tuner.get_best_models(num_models=2)
best_model = models[0]
best_model.summary()

# 결과 요약
tuner.results_summary()

# 최적의 하이퍼파라미터
best_hps = tuner.get_best_hyperparameters(5)

model = build_model(best_hps[0])

x_all = np.concatenate((x_train, x_val))
y_all = np.concatenate((y_train, y_val))
model.fit(x=x_all, y=y_all, epochs=1)

# 커스텀 하이퍼모델 클래스
class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        model.add(layers.Flatten())
        model.add(
            layers.Dense(
                units=hp.Int("units", min_value=32, max_value=512, step=32),
                activation="relu",
            )
        )
        model.add(layers.Dense(10, activation="softmax"))
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            shuffle=hp.Boolean("shuffle"),
            **kwargs,
        )

# 커스텀 하이퍼모델 사용
hp = keras_tuner.HyperParameters()
hypermodel = MyHyperModel()
model = hypermodel.build(hp)
hypermodel.fit(hp, model, np.random.rand(100, 28, 28), np.random.rand(100, 10))

# 커스텀 하이퍼모델
class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        model.add(layers.Flatten())
        model.add(
            layers.Dense(
                units=hp.Int("units", min_value=32, max_value=512, step=32),
                activation="relu",
            )
        )
        model.add(layers.Dense(10, activation="softmax"))
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # 커스텀 학습 메소드
    def fit(self, hp, model, x, y, **kwargs):
        if hp.Boolean("normalize"):
            x = layers.Normalization()(x)
        return model.fit(
            x,
            y,
            shuffle=hp.Boolean("shuffle"),
            **kwargs,
        )

# 커스텀 하이퍼모델 사용
hp = keras_tuner.HyperParameters()
hypermodel = MyHyperModel()
model = hypermodel.build(hp)
hypermodel.fit(hp, model, np.random.rand(100, 28, 28), np.random.rand(100, 10))

# 커스텀 하이퍼모델 정의
class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        image_size = hp.Int("image_size", 10, 28)
        inputs = keras.Input(shape=(image_size, image_size))
        outputs = layers.Flatten()(inputs)
        outputs = layers.Dense(
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
        )(outputs)
        outputs = layers.Dense(10, activation="softmax")(outputs)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
    
     # 커스텀 학습 메소드
    def fit(self, hp, model, x, y, validation_data=None, **kwargs):
        if hp.Boolean("normalize"):
            x = layers.Normalization()(x)
        image_size = hp.get("image_size")
        cropped_x = x[:, :image_size, :image_size, :]
        if validation_data:
            x_val, y_val = validation_data
            cropped_x_val = x_val[:, :image_size, :image_size, :]
            validation_data = (cropped_x_val, y_val)
        return model.fit(
            cropped_x,
            y,
            shuffle=hp.Boolean("shuffle"),
            validation_data=validation_data,
            **kwargs,
        )

# 하이퍼모델을 사용한 하이퍼파라미터 튜닝
tuner = keras_tuner.RandomSearch(
    MyHyperModel(),
    objective="val_accuracy",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="tune_hypermodel",
)

# 하이퍼튜닝 시작
tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))

# 모델 빌드 및 학습
hypermodel = MyHyperModel()
best_hp = tuner.get_best_hyperparameters()[0]
model = hypermodel.build(best_hp)
hypermodel.fit(best_hp, model, x_all, y_all, epochs=1)

# 회귀 모델 빌드 함수
def build_regressor(hp):
    model = keras.Sequential(
        [
            layers.Dense(units=hp.Int("units", 32, 128, 32), activation="relu"),
            layers.Dense(units=1),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        metrics=[keras.metrics.MeanAbsoluteError()],
    )
    return model

# 회귀 모델 하이퍼튜닝 설정
tuner = keras_tuner.RandomSearch(
    hypermodel=build_regressor,
    objective=keras_tuner.Objective("val_mean_absolute_error", direction="min"),
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="built_in_metrics",
)

# 하이퍼튜닝 실행
tuner.search(
    x=np.random.rand(100, 10),
    y=np.random.rand(100, 1),
    validation_data=(np.random.rand(20, 10), np.random.rand(20, 1)),
)

# 하이퍼튜닝 결과 요약
tuner.results_summary()

from keras import ops

class CustomMetric(keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(name="custom_metric", **kwargs)
        self.sum = self.add_weight(name="sum", initializer="zeros")
        self.count = self.add_weight(name="count", dtype="int32", initializer="zeros")

    # 상태 업데이트
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        self.sum.assign_add(keras.backend.mean(keras.backend.abs(y_true - y_pred)))
        self.count.assign_add(1)

    # 메트릭 결과 반환
    def result(self):
        return self.sum / self.count

     # 상태 리셋
    def reset_states(self):
        self.sum.assign(0)
        self.count.assign(0)

    # 메트릭 구성 반환
    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

# 모델 빌드
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
        )
    )
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", CustomMetric()],
    )
    return model
