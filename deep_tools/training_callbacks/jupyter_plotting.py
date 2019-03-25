from IPython.display import clear_output
from keras.callbacks import Callback
from matplotlib import pyplot as plt

from deep_tools.training_callbacks import TestLossAccCallback

# Plotting callbacks for jupyter notebook executions

def reset_jupyter_plotting_instances():
    AbstractEpochPlotter.reset_num_of_instances()

class AbstractEpochPlotter(Callback):
    num_of_instances = 0

    @staticmethod
    def reset_num_of_instances():
        AbstractEpochPlotter.num_of_instances = 0

    def __init__(self):
        super(AbstractEpochPlotter, self).__init__()
        AbstractEpochPlotter.num_of_instances += 1
        self.figure_num = AbstractEpochPlotter.num_of_instances


class EpochPlotter(AbstractEpochPlotter):
    def __init__(self, title="", label="", **labeled_data):
        """
        Custom callback plotter that refreshes plots on epoch end in Jupyter notebook
        :param title: title of the plot
        :param label: ylable of the plot
        :param labeled_data: dictionary of structure: label(str) : data(iterable)
        """

        super(EpochPlotter, self).__init__()
        self.epochs_x = []

        self.title = title
        self.ylabel = label
        self.labeled_data = labeled_data

    def _plot(self, figure_num, title, ylabel, **labeled_plots):
        if figure_num == 1:
            clear_output(wait=True)
        plt.figure(figure_num)
        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)

        for label, data in labeled_plots.items():
            plt.plot(self.epochs_x, data, label=label)
        plt.legend()
        plt.show()

    def on_train_begin(self, logs={}):
        self.epochs_x = []

    def on_epoch_end(self, epoch, logs={}):
        self.epochs_x.append(epoch)


        clear_output(wait=True)

        self._plot(self.figure_num, self.title, self.ylabel, **self.labeled_data)

    @property
    def history(self):
        return self.labeled_data


class TrainingLossPlotter(EpochPlotter):
    def __init__(self):
        """
        Custom callback plotter that refreshes plots of loss on epoch end in Jupyter notebook
        """
        super(TrainingLossPlotter, self).__init__("Training Loss", "loss", loss=[], val_loss=[])

    def on_epoch_end(self, epoch, logs={}):
        self.labeled_data["loss"].append(logs.get('loss'))
        self.labeled_data["val_loss"].append(logs.get('val_loss'))
        super(TrainingLossPlotter, self).on_epoch_end(epoch, logs)


class TrainingAccuracyPlotter(EpochPlotter):
    def __init__(self):
        """
        Custom callback plotter that refreshes plots of accuracy on epoch end in Jupyter notebook
        """
        super(TrainingAccuracyPlotter, self).__init__("Training Accuracy", "accuracy", acc=[], val_acc=[])

    def on_epoch_end(self, epoch, logs={}):
        self.labeled_data["acc"].append(logs.get('acc'))
        self.labeled_data["val_acc"].append(logs.get('val_acc'))
        super(TrainingAccuracyPlotter, self).on_epoch_end(epoch, logs)


class TestingAccuracyPlotter(TestLossAccCallback, EpochPlotter):
    def __init__(self, xtest, ytest):
        """
        Custom callback plotter that refreshes plots of loss on epoch end in Jupyter notebook
        """
        TestLossAccCallback.__init__(self, xtest, ytest, True)
        labeled_data = dict()
        labeled_data["acc"] = self.acc_history

        EpochPlotter.__init__(self, "Accuracy On The Test Data", "accuracy", **labeled_data)

    def on_epoch_end(self, epoch, logs={}):
        TestLossAccCallback.on_epoch_end(self, epoch, logs)
        EpochPlotter.on_epoch_end(self, epoch, logs)

