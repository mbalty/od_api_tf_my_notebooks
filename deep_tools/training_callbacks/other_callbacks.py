import datetime
from time import time

from keras.callbacks import ModelCheckpoint, Callback

from deep_tools.general import timeDifSecondsToStr


class TestLossAccCallback(Callback):
    def __init__(self, xtest, ytest, print_test_acc_loss=False):
        super(TestLossAccCallback, self).__init__()
        self.xtest = xtest
        self.ytest = ytest
        self.loss_history = []
        self.acc_history = []
        self.print_test_acc_loss = print_test_acc_loss

    def on_epoch_end(self, epoch, logs={}):
        print ("debug ", self.xtest.shape)
        print ("debug ", self.ytest.shape)
        loss, acc = self.model.evaluate(self.xtest, self.ytest, verbose=0)
        self.loss_history.append(loss)
        self.acc_history.append(acc)
        if self.print_test_acc_loss:
            print('\nTesting loss: {}, acc: {}'.format(loss, acc))


def checkpoint_classification_callback(checkpoint_path,
                                       naming_rule="weights-epoch{epoch:02d}-val_acc{val_acc:.2f}-val_loss{val_loss:.2f}.hdf5",
                                       save_best_only=True,
                                       monitor='val_acc'):
    """
    Model checkpoint callback.
    :param checkpoint_path:
    :param naming_rule:
    :param save_best_only:
    :param monitor:
    :return:
    """
    filepath = (checkpoint_path + "/"
                + str(datetime.datetime.now().date())
                + naming_rule)
    return ModelCheckpoint(filepath, monitor=monitor, verbose=1, save_best_only=save_best_only, mode='max')


class TimeLogCallback(Callback):
    """
    logs total training time
    """
    def __init__(self):
        super(TimeLogCallback, self).__init__()
        self.start_time = 0
        self.current_time = 0

    def on_train_begin(self, logs=None):
        self.start_time = time()
        self.current_time = time()

    def on_epoch_end(self, epoch, logs=None):
        self.current_time = time()
        print("Total training time: " + timeDifSecondsToStr(self.total_time))

    @property
    def total_time(self):
        return self.current_time - self.start_time