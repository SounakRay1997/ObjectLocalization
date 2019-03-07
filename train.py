import csv
import math

from PIL import Image
import numpy as np
from keras.models import load_model
import tensorflow as tf
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from keras.layers import Conv2D, Reshape, Dense, GlobalAveragePooling2D, MaxPooling2D, Activation, Input
from keras.applications.mobilenet import preprocess_input
from keras.utils import Sequence
from keras.optimizers import Adam
from keras.backend import epsilon

# 0.35, 0.5, 0.75, 1.0
ALPHA = 0.35

# 96, 128, 160, 192, 224
IMAGE_SIZE = 224

EPOCHS = 100
BATCH_SIZE = 5
PATIENCE = 20
image_width = 640.0
image_height = 480.0

TRAIN_CSV = "training.csv"
VALIDATION_CSV = "validation.csv"

class DataGenerator(Sequence):

    def __init__(self, csv_file):
        self.paths = []

        with open(csv_file, "r") as file:
            self.coords = np.zeros((sum(1 for line in file), 4))
            file.seek(0)

            reader = csv.reader(file, delimiter=",")
            for index, row in enumerate(reader):
                path, x1, x2, y1, y2 = row
                x1=np.float32(x1)
                x2=np.float32(x2)
                y1=np.float32(y1)
                y2=np.float32(y2)
                self.coords[index, 0] = (x1 * IMAGE_SIZE) / image_width
                self.coords[index, 1] = (x2 * IMAGE_SIZE) / image_width
                self.coords[index, 2] = (y1 * IMAGE_SIZE) / image_height
                self.coords[index, 3] = (y2 * IMAGE_SIZE) / image_height
                path1='/home/sounak/FlipkartGRiDLevel3/train/'+path   
                self.paths.append(path1)

    def __len__(self):
        return math.ceil(len(self.coords) / BATCH_SIZE)

    def __getitem__(self, idx):
        batch_paths = self.paths[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
        batch_coords = self.coords[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]

        batch_images = np.zeros((len(batch_paths), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
        for i, f in enumerate(batch_paths):
            img = Image.open(f)
            img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
            img = img.convert('RGB')

            batch_images[i] = preprocess_input(np.array(img, dtype=np.float32))
            img.close()

        return batch_images, batch_coords

class Validation(Callback):
    def __init__(self, generator):
        self.generator = generator

    def on_epoch_end(self, epoch, logs):
        mse = 0

        intersections = 0
        unions = 0

        for i in range(len(self.generator)):
            batch_images, gt = self.generator[i]
            pred = self.model.predict_on_batch(batch_images)
            mse += np.linalg.norm(gt - pred, ord='fro') / pred.shape[0]

            pred = np.maximum(pred, 0)

            diff_width = np.minimum(gt[:,0] + gt[:,1], pred[:,0] + pred[:,1]) - np.maximum(gt[:,0], pred[:,0])
            diff_height = np.minimum(gt[:,2] + gt[:,3], pred[:,2] + pred[:,3]) - np.maximum(gt[:,2], pred[:,2])
            intersection = np.maximum(diff_width, 0) * np.maximum(diff_height, 0)

            area_gt = gt[:,1] * gt[:,3]
            area_pred = pred[:,1] * pred[:,3]
            union = np.maximum(area_gt + area_pred - intersection, 0)

            intersections += np.sum(intersection * (union > 0))
            unions += np.sum(union)

        iou = np.round(intersections / (unions + epsilon()), 4)
        logs["val_iou"] = iou

        mse = np.round(mse, 4)
        logs["val_mse"] = mse


        print(" - val_iou: {} - val_mse: {}".format(iou, mse))


def create_model():
    img_input = Input((224, 224, 3))
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(img_input)
    x = Activation('relu')(x)
    x = MaxPooling2D((4, 4), strides=(2, 2))(x)
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((4, 4), strides=(2, 2))(x)
    x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((4, 4), strides=(2, 2))(x)
    x = Conv2D(32, kernel_size=(2, 2), strides=(2, 2))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((4, 4), strides=(2, 2))(x)
    x = Conv2D(4, kernel_size=(2, 2))(x)
    x = Reshape((4,), name="coords")(x)
    return Model(inputs=img_input, outputs=x)


def log_mse(y_true, y_pred):
    return tf.reduce_mean(tf.log1p(tf.squared_difference(y_pred, y_true)), axis=-1)

def main():
    model=load_model('/home/sounak/FlipkartGRIDNew/model-0.87_Final.h5', custom_objects={'log_mse':log_mse})

    train_datagen = DataGenerator(TRAIN_CSV)
    validation_datagen = Validation(generator=DataGenerator(VALIDATION_CSV))

    optimizer = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=log_mse, optimizer=optimizer, metrics=[])
    checkpoint = ModelCheckpoint("model-{val_iou:.2f}_3.h5", monitor="val_iou", verbose=1, save_best_only=True, mode="max")
    stop = EarlyStopping(monitor="val_iou", patience=PATIENCE, mode="max")
    reduce_lr = ReduceLROnPlateau(monitor="val_iou", factor=0.2, patience=10, min_lr=1e-7, verbose=1, mode="max")

    model.summary()

    model.fit_generator(generator=train_datagen, epochs=EPOCHS, callbacks=[validation_datagen, checkpoint, reduce_lr, stop])


if __name__ == "__main__":
    main()
