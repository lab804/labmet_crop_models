import numpy as np
import neurolab as nl
import os
import pickle
import pylab as pl

from crop_models.ann.config import *
from crop_models.ann.annexceptions import *


class TimeSeriesMLPMultivariate(object):
    """ Time Series Multilayer Perceptron Ann

    This class is responsible for wrapping and acting
    as a proxy to the neurolab's multilayer perceptron function and
    its correlated pseudo methods.

    """

    # __base_dir = os.path.dirname(os.path.abspath(__file__)).split("labmet_ann")[0]

    def __init__(self, topology,
                 train_alg="train_gdx", error_function="mse"):

        self.hidden_layers = topology

        if train_alg not in mlp_train_algorithm:
            raise (TrainAlgorithmException("This is not an valid Train Algorithm"))
        else:
            self.train_alg = train_alg

        if error_function not in error_functions:
            raise (TrainAlgorithmException("This is not an valid error function"))
        else:
            self.error_function = error_function

        self.ann = None
        self._train_data = []
        self._target_data = []

    @classmethod
    def load(cls, ann_filename):
        """

        :param ann_filename:
        :return:
        """
        f_name = "{}.pkl".format(ann_filename)
        if os.path.exists(f_name):
            with open("{}.pkl".format(ann_filename), "rb") as ann_file:
                ann = pickle.load(ann_file)
            if not isinstance(ann, cls):
                raise AnnFileTypeException("This is not an valid "
                                           "TimeSeriesMLPMultivariate file type")
            return ann
        else:
            raise FileExistsError("The ann file was't found")

    @staticmethod
    def __train_data_range(data):
        """

        :param data:

        :type data: list or ndarray
        :return:
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        return [[np.min(i), np.max(i)] for i in data.transpose()]

    @staticmethod
    def __plot(*args, save_plot=False, plot_type='plot', **kwargs):
        if plot_type == 'plot':
            pl.plot(*args)
        elif plot_type == 'scatter':
            pl.scatter(*args)
        pl.xlabel(kwargs["x_label"]),
        pl.ylabel(kwargs["y_label"])
        if "title" in kwargs:
            pl.title(kwargs['title'])
        if save_plot:
            try:
                pl.savefig(fname=kwargs["filename"])
            except:
                pl.savefig(filename=kwargs["filename"])
        if "show" in kwargs:
            pl.show()
        else:
            pl.close()

    def __ann(self, data_range_matrix):
        """

        :return:
        """
        ann = nl.net.newff(minmax=data_range_matrix,
                           size=self.hidden_layers)

        ann.errorf = error_functions[self.error_function]
        ann.trainf = mlp_train_algorithm[self.train_alg]
        #todo deixar isso como opcao.
        for l in ann.layers:
            # l.transf = nl.trans.SoftMax()
            # l.transf = nl.trans.SatLins()
            # l.transf = nl.trans.LogSig()
            l.transf = nl.trans.TanSig()
            # l.initf = nl.init.InitRand([0.001, 0.002], 'wb')
            l.initf = nl.init.InitRand([-0.01, 0.02], 'wb')

        self.ann = ann
        return ann

    def train(self, train_set, show=1, plot=True, save_plot=False, filename="train_plt", **kwargs):
        """

        :param train_set:
        :param show:
        :param plot:
        :param save_plot:
        :param kwargs:
        :return:
        """
        self._train_data = []
        self._target_data = []
        for i in train_set:
            self._train_data.append(i[0])
            self._target_data.append(i[1])

        range_matrix = self.__train_data_range(self._train_data)


        self.ann = self.__ann(range_matrix)
        error_matrix = self.ann.train(input=self._train_data,
                                      target=self._target_data,
                                      show=show,
                                      **kwargs)
        if plot:
            self.__plot(error_matrix,
                        x_label='Epoch number(Numero da epoca)',
                        y_label='error function valeu({})'.format(self.error_function.upper()),
                        save_plot=save_plot,
                        filename=filename,
                        **kwargs)

        return error_matrix

    def sim(self, save_plot=False, plot=True, plot_type='scatter', filename="train_plt", **kwargs):


        if self.ann is None:
            raise AnnTrainException("It's required to train an Artificial Neural Network")
        if len(self._train_data) == 0:
            raise AnnTrainException("It's required a train set for simulating outputs")

        sim_values = self.ann.sim(self._train_data)
        x = range(0, len(sim_values))
        if plot:
            if plot_type == 'plot':
                self.__plot(x, sim_values, x, self._target_data, plot_type='plot',
                            save_plot=save_plot,
                            filename=filename,
                            **kwargs)
            elif plot_type == 'scatter':
                self.__plot(sim_values, self._target_data, plot_type='scatter',
                            save_plot=save_plot,
                            filename=filename,
                            **kwargs)
        return sim_values

    def out(self, real_world_data, save_plot=False, plot=True, plot_type='scatter', filename="train_plt", **kwargs):
        """

        :return:
        """
        # input_train_data = []
        # validation_data = []
        # for i in real_world_data:
        #     input_train_data.append(i[0])
        #     validation_data.append(i[1])

        out_values = self.ann.sim(real_world_data)
        # x = range(0, len(out_values))
        if plot:
            if plot_type == 'plot':
                self.__plot(kwargs["x"], out_values, plot_type='plot',
                            save_plot=save_plot,
                            filename=filename,
                            **kwargs)
            elif plot_type == 'scatter':
                self.__plot(kwargs["x"], out_values, plot_type='scatter',
                            save_plot=save_plot,
                            filename=filename,
                            **kwargs)

        return out_values

    def __pickle_helper(self, f_name):
        """

        :param f_name:
        :return:
        """
        with open(f_name, "wb") as ann_file:
            pickle.dump(self, ann_file, pickle.HIGHEST_PROTOCOL)

    def save(self, ann_filename):
        """

        :param ann_filename:
        :return:
        """
        f_name = "{}.pkl".format(ann_filename)
        if not os.path.exists( f_name):
            self.__pickle_helper(f_name)
        else:
            f_name_index = 1
            new_f_name = "{}_copy_{}.pkl".format(ann_filename, f_name_index)

            while os.path.exists(new_f_name):
                f_name_index += 1
                new_f_name = "{}_copy_{}.pkl".format(ann_filename, f_name_index)
            self.__pickle_helper(new_f_name)

class TimeSeriesMLPMultivariateTemp(object):
    """ Time Series Multilayer Perceptron Ann

    This class is responsible for wrapping and acting
    as a proxy to the neurolab's multilayer perceptron function and
    its correlated pseudo methods.

    """

    __base_dir = os.path.dirname(os.path.abspath(__file__)).split("labmet_ann")[0]

    def __init__(self, train_data, validation_data,
                 target_train_data, target_validation_data, hidden_layers,
                 train_alg="train_gdx", error_function="mse"):
        """

        :param train_data:
        :param validation_data:
        :param target_train_data:
        :param target_validation_data:
        :param hidden_layers:
        :param train_alg:
        :param error_function:
        """

        self.train_data = train_data
        self.validation_data = validation_data
        self.target_train_data = target_train_data
        self.target_validation_data = target_validation_data
        self.hidden_layers = hidden_layers

        if train_alg not in mlp_train_algorithm:
            raise (TrainAlgorithmException("This is not an valid Train Algorithm"))
        else:
            self.train_alg = train_alg

        if error_function not in error_functions:
            raise (TrainAlgorithmException("This is not an valid error function"))
        else:
            self.error_function = error_function

        self.ann = self.__ann()

    def __train_data_range(self):
        """

        :return:
        """
        return [[np.min(i), np.max(i)] for i in self.train_data.transpose()]

    def __ann(self):
        """

        :return:
        """
        ann = nl.net.newff(minmax=self.__train_data_range(),
                           size=self.hidden_layers)
        ann.errorf = error_functions[self.error_function]
        ann.trainf = mlp_train_algorithm[self.train_alg]
        for l in ann.layers:
            l.initf = nl.init.InitRand([-0.05, 0.05], 'wb')
        return ann

    def __pickle_helper(self, f_name):
        """

        :param f_name:
        :return:
        """
        with open(f_name, "wb") as ann_file:
            pickle.dump(self, ann_file, pickle.HIGHEST_PROTOCOL)

    def train(self, show=1, **kwargs):
        """

        :param show:
        :param kwargs:
        :return:
        """
        return self.ann.train(input=self.train_data,
                              target=self.target_train_data,
                              show=show,
                              **kwargs)

    def out(self):
        """

        :return:
        """
        return self.ann.sim(self.train_data)

    def save(self, ann_filename):
        """

        :param ann_filename:
        :return:
        """
        f_name = "{}.pkl".format(ann_filename)
        if not os.path.exists(os.path.join(self.__base_dir, f_name)):
            self.__pickle_helper(f_name)
        else:
            f_name_index = 1
            new_f_name = "{}_copy_{}.pkl".format(ann_filename, f_name_index)

            while os.path.exists(os.path.join(self.__base_dir, new_f_name)):
                f_name_index += 1
                new_f_name = "{}_copy_{}.pkl".format(ann_filename, f_name_index)
            self.__pickle_helper(new_f_name)

    @classmethod
    def load(cls, ann_filename):
        """

        :param ann_filename:
        :return:
        """
        f_name = "{}.pkl".format(ann_filename)
        if os.path.exists(os.path.join(cls.__base_dir, f_name)):
            with open("{}.pkl".format(ann_filename), "rb") as ann_file:
                ann = pickle.load(ann_file)
            if not isinstance(ann, cls):
                raise AnnFileTypeException("This is not an valid "
                                           "TimeSeriesMLPMultivariate file type")
            return ann
        else:
            raise FileExistsError("The ann file was't found")

