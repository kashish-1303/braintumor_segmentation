# #final code
# import numpy as np
# import random
# import json
# import os
# from glob import glob
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
# import tensorflow.keras.backend as K
# from tensorflow.keras.optimizers import SGD
# from tensorflow.keras.optimizers.schedules import ExponentialDecay
# from model import Unet_model
# from losses import *
# from sklearn.model_selection import train_test_split

# # Create directory for saving models if it doesn't exist
# if not os.path.exists('brain_segmentation'):
#     os.makedirs('brain_segmentation')

# class SGDLearningRateTracker(Callback):
#     def on_epoch_begin(self, epoch, logs=None):
#         optimizer = self.model.optimizer
#         lr = optimizer.learning_rate.numpy()
#         print('Current LR:', lr)

# class Training(object):
#     def __init__(self, batch_size, nb_epoch, load_model_resume_training=None):
#         self.batch_size = batch_size
#         self.nb_epoch = nb_epoch

#         if load_model_resume_training is not None:
#             self.model = self.load_model(load_model_resume_training)
#             print("pre-trained model loaded!")
#         else:
#             unet = Unet_model(img_shape=(128, 128, 4))
#             self.model = unet.model
            
#             # Define learning rate schedule
#             initial_learning_rate = 0.01
#             lr_schedule = ExponentialDecay(
#                 initial_learning_rate,
#                 decay_steps=100000,
#                 decay_rate=0.96,
#                 staircase=True)
            
#             # Compile the model with the learning rate schedule
#             self.model.compile(optimizer=SGD(learning_rate=lr_schedule),
#                                loss=gen_dice_loss,
#                                metrics=[dice_whole_metric, dice_core_metric, dice_en_metric])
            
#             print("U-net CNN compiled!")

#     def fit_unet(self, X_train, Y_train, X_valid, Y_valid):
#         train_generator = self.img_msk_gen(X_train, Y_train, 9999)
#         checkpointer = ModelCheckpoint(
#             filepath='brain_segmentation/ResUnet.{epoch:02d}_{val_loss:.3f}.keras', 
#             verbose=1,
#             save_best_only=True
#         )
#         self.model.fit(
#             train_generator,
#             steps_per_epoch=len(X_train) // self.batch_size,
#             epochs=self.nb_epoch,
#             validation_data=(X_valid, Y_valid),
#             verbose=1,
#             callbacks=[checkpointer, SGDLearningRateTracker()]
#         )

#     def img_msk_gen(self, X_train, Y_train, seed):
#         datagen = ImageDataGenerator(horizontal_flip=True, data_format="channels_last")
#         datagen_msk = ImageDataGenerator(horizontal_flip=True, data_format="channels_last")
#         image_generator = datagen.flow(X_train, batch_size=self.batch_size, seed=seed)
#         y_generator = datagen_msk.flow(Y_train, batch_size=self.batch_size, seed=seed)
#         while True:
#             X_batch = next(image_generator)
#             Y_batch = next(y_generator)
#             yield X_batch, Y_batch

#     def save_model(self, model_name):
#         self.model.save(f'{model_name}.keras')
#         print('Model saved.')

#     def load_model(self, model_name):
#         print(f'Loading model {model_name}')
#         model = load_model(model_name, custom_objects={
#             'gen_dice_loss': gen_dice_loss,
#             'dice_whole_metric': dice_whole_metric,
#             'dice_core_metric': dice_core_metric,
#             'dice_en_metric': dice_en_metric
#         })
#         print('Model loaded.')
#         return model

# if __name__ == "__main__":
#     # Set arguments
#     brain_seg = Training(batch_size=4, nb_epoch=1, load_model_resume_training=None)

#     print("number of trainable parameters:", brain_seg.model.count_params())

#     # Load data from disk
#     X_patches = np.load("x_training.npy").astype(np.float32)
#     Y_labels = np.load("y_training.npy").astype(np.uint8)
#     print("loading patches done\n")

#     # Split data into training and validation sets
#     X_train, X_valid, Y_train, Y_valid = train_test_split(
#         X_patches, Y_labels, test_size=0.2, random_state=42
#     )

#     print("Training data shape:", X_train.shape)
#     print("Training labels shape:", Y_train.shape)
#     print("Validation data shape:", X_valid.shape)
#     print("Validation labels shape:", Y_valid.shape)

#     # Fit model
#     brain_seg.fit_unet(X_train, Y_train, X_valid, Y_valid)

# import numpy as np
# from model import TwoPathwayGroupCNN
# from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# class Training(object):
#     def __init__(self, batch_size, nb_epoch, load_model_resume_training=None):
#         self.batch_size = batch_size
#         self.nb_epoch = nb_epoch
#         self.load_model_resume_training = load_model_resume_training
#         self.model = self.get_model()

#     def get_model(self):
#         model = TwoPathwayGroupCNN(img_shape=(128, 128, 4), 
#                                    load_model_weights=self.load_model_resume_training)
#         return model.model

#     def fit_unet(self, X_train, Y_train, X_valid, Y_valid):
#         train_generator = self.img_msk_gen(X_train, Y_train, 9999)
#         checkpointer = ModelCheckpoint(
#             filepath='brain_segmentation/TwoPathwayGroupCNN.{epoch:02d}_{val_loss:.3f}.keras', 
#             verbose=1,
#             save_best_only=True
#         )
        
#         lr_scheduler = LearningRateScheduler(self.lr_schedule)
        
#         self.model.fit(
#             train_generator,
#             steps_per_epoch=len(X_train) // self.batch_size,
#             epochs=self.nb_epoch,
#             validation_data=(X_valid, Y_valid),
#             verbose=1,
#             callbacks=[checkpointer, lr_scheduler]
#         )

#     def img_msk_gen(self, X_train, Y_train, seed):
#         datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
#         datagen_msk = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
#         image_generator = datagen.flow(X_train, batch_size=self.batch_size, seed=seed)
#         mask_generator = datagen_msk.flow(Y_train, batch_size=self.batch_size, seed=seed)
#         while True:
#             yield (next(image_generator), next(mask_generator))

#     def lr_schedule(self, epoch):
#         lr = 1e-3
#         if epoch > 180:
#             lr *= 0.5e-3
#         elif epoch > 150:
#             lr *= 1e-3
#         elif epoch > 120:
#             lr *= 1e-2
#         elif epoch > 80:
#             lr *= 1e-1
#         print('Learning rate: ', lr)
#         return lr

# # Main execution
# if __name__ == "__main__":
#     X_patches = np.load("x_training.npy").astype(np.float32)
#     Y_labels = np.load("y_training.npy").astype(np.uint8)
    
#     from sklearn.model_selection import train_test_split
#     X_train, X_valid, Y_train, Y_valid = train_test_split(X_patches, Y_labels, test_size=0.2, random_state=42)
#     print("X_train shape:", X_train.shape)
#     print("Y_train shape:", Y_train.shape)
    
#     brain_seg = Training(batch_size=32, nb_epoch=200)
#     brain_seg.fit_unet(X_train, Y_train, X_valid, Y_valid)

import numpy as np
from model import TwoPathwayGroupCNN
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

class Training(object):
    def __init__(self, batch_size, nb_epoch, load_model_resume_training=None):
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.load_model_resume_training = load_model_resume_training
        self.model = self.get_model()

    def get_model(self):
        model = TwoPathwayGroupCNN(img_shape=(128, 128, 4), 
        load_model_weights=self.load_model_resume_training)
        return model.model

    def fit_unet(self, X_train, Y_train, X_valid, Y_valid):
        print("Preparing data generator...")
        train_generator = self.img_msk_gen(X_train, Y_train, 9999)
        print("Setting up callbacks...")
        checkpointer = ModelCheckpoint(
            filepath='brain_segmentation/TwoPathwayGroupCNN.{epoch:02d}_{val_loss:.3f}.keras', 
            verbose=1,
            save_best_only=True
        )
        
        lr_scheduler = LearningRateScheduler(self.lr_schedule)
        print("Starting model fit...")
        print("Starting model training...")
        self.model.fit(
            train_generator,
            steps_per_epoch=len(X_train) // self.batch_size,
            epochs=self.nb_epoch,
            validation_data=(X_valid, Y_valid),
            verbose=1,
            callbacks=[checkpointer, lr_scheduler]
        )
        print("Model training completed.")

    def img_msk_gen(self, X_train, Y_train, seed):
        datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
        datagen_msk = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
        image_generator = datagen.flow(X_train, batch_size=self.batch_size, seed=seed)
        mask_generator = datagen_msk.flow(Y_train, batch_size=self.batch_size, seed=seed)
        while True:
            yield (next(image_generator), next(mask_generator))

    def lr_schedule(self, epoch):
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 150:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate: ', lr)
        return lr

if __name__ == "__main__":
    
    X_patches = np.load("x_training.npy").astype(np.float32)
    Y_labels = np.load("y_training.npy").astype(np.float32)
    N= 16
    X_patches=X_patches[:N]
    Y_labels=Y_labels[:N]
    
    print("X_train shape:", X_patches.shape)
    print("Y_train shape:", Y_labels.shape)
    
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_patches, Y_labels, test_size=0.2, random_state=42)

    print("after split")
    print("x",X_train.shape)
    print("x",Y_train.shape)
    print("x",Y_valid.shape)
    print("x",X_valid.shape)
    
    brain_seg = Training(batch_size=2, nb_epoch=1)
    brain_seg.fit_unet(X_train, Y_train, X_valid, Y_valid)

