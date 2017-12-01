import numpy as np
import neurolab as nl
import os
import pickle

from crop_models.ann.config import *
from crop_models.ann.annexceptions import *


class TimeSeriesMLPMultivariate(object):
    """ Time Series Multilayer Perceptron Ann

    This class is responsible for wrapping and acting
    as a proxy to the neurolab's multilayer perceptron function and
    its correlated pseudo methods.

    """

    # __base_dir = os.path.dirname(os.path.abspath(__file__)).split("labmet_ann")[0]

    def __init__(self, topology, data_range_matrix,
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

        self.data_range_matrix = np.array(data_range_matrix)
        self.ann = self.__ann()

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

    def __ann(self):
        """

        :return:
        """
        ann = nl.net.newff(minmax=self.data_range_matrix,
                           size=self.hidden_layers)

        ann.errorf = error_functions[self.error_function]
        ann.trainf = mlp_train_algorithm[self.train_alg]

        for l in ann.layers:
            l.initf = nl.init.InitRand([0.001, 0.8], 'wb')

        return ann

    def train(self,train_data, target_data, show=1, **kwargs):
        """

        :param show:
        :param kwargs:
        :return:
        """
        return self.ann.train(input=train_data,
                              target=target_data,
                              show=show,
                              **kwargs)

    def out(self, real_world_data):
        """

        :return:
        """
        return self.ann.sim(real_world_data)


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
            l.initf = nl.init.InitRand([0.001, 0.8], 'wb')
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

