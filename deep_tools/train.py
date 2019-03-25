import os
import pickle
from keras.callbacks import TensorBoard

from deep_tools.training_callbacks import *


class Trainer:
    """
    Trainer class used for handling callbacks and training both a jupyter notebook and standalone python.
    It eases training sequences. saves the  training configuration, so that on can continue training.
    """
    def __init__(self):
        self.model = None
        self.model_name = None
        self.Xtrain = None
        self.Ytrain = None
        self.Xtest = None
        self.Ytest = None
        self.checkpoint_location = None
        self.epoch = None
        self.callbacks_list = None
        self.count_run = 0

    def train(self, model, model_name, images_path, checkpoint_location, image_generation_function, checkpoint_naming_rule=None, epochs=10,
              tensorboard_callback=True, tensorboard_log_directory=None,
              image_augmentation_factor=9, extra_callbacks=list()):

        self.model = model
        self.model_name = model_name
        self.epoch = epochs
        self.count_run += 1

        self.Xtrain, self.Ytrain, self.Xtest, self.Ytest = \
                                        image_generation_function(images_path,
                                        n_augmented_extra=image_augmentation_factor,
                                        test_cut=.2, resize_side=model.input.shape[1])
        print("Training...")

        # add callbacks
        full_checkpoint_path = checkpoint_location + "/" + model_name
        self.checkpoint_location = full_checkpoint_path

        if not os.path.exists(full_checkpoint_path):
            os.makedirs(full_checkpoint_path)

        if checkpoint_naming_rule is None:
            checkpoint = checkpoint_classification_callback(full_checkpoint_path, save_best_only=True)
        else:
            checkpoint = checkpoint_classification_callback(full_checkpoint_path, naming_rule=checkpoint_naming_rule, save_best_only=True)


        self.callbacks_list = [checkpoint] + extra_callbacks

        time_callback = TimeLogCallback()
        self.callbacks_list.append(time_callback)


        if (tensorboard_callback):
            full_tensorboard_log_path = tensorboard_log_directory + "/" + model_name
            if not os.path.exists(full_tensorboard_log_path):
                os.makedirs(full_tensorboard_log_path)
            tensorboard = TensorBoard(log_dir=full_tensorboard_log_path,
                                      histogram_freq=1, batch_size=32, write_graph=False, write_grads=False,
                                      write_images=False, embeddings_freq=0, embeddings_layer_names=None,
                                      embeddings_metadata=None)

            self.callbacks_list.append(tensorboard)

        history = self.model.fit(self.Xtrain, self.Ytrain,
                                 epochs=epochs,
                                 validation_data=(self.Xtest, self.Ytest),
                                 callbacks=self.callbacks_list,
                                 batch_size=32,
                                 verbose=1)

        pickle.dump(history.history,
                    open(full_checkpoint_path + "/training_history.p", "wb"))

        return self.model

    def jupyter_plotting_train(self, model, model_name, images_path, checkpoint_location, image_generation_function,
                               checkpoint_naming_rule=None, epochs=10,
                               plot_training_loss=True, plot_training_acc=True, tensorboard_callback=True,
                               tensorboard_log_directory=None, image_augmentation_factor=9):

        reset_jupyter_plotting_instances()

        callbacks_list = []

        if plot_training_loss:
            plot_loss = TrainingLossPlotter()
            callbacks_list.append(plot_loss)

        if plot_training_acc:
            plot_acc = TrainingAccuracyPlotter()

            callbacks_list.append(plot_acc)

        return self.train(model, model_name, images_path, checkpoint_location, image_generation_function,
                          checkpoint_naming_rule, epochs, tensorboard_callback,
                          tensorboard_log_directory, image_augmentation_factor, extra_callbacks=callbacks_list)

    def continue_training(self, epochs=10):
        self.count_run += 1

        history = self.model.fit(self.Xtrain, self.Ytrain,
                                 epochs=epochs,
                                 initial_epoch=self.epoch,
                                 validation_data=(self.Xtest, self.Ytest),
                                 callbacks=self.callbacks_list,
                                 batch_size=32,
                                 verbose=1)

        pickle.dump(history.history,
                    open(self.checkpoint_location + "/training_history_" + str(self.count_run) + ".p", "wb"))

        return self.model


