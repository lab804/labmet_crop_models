from itertools import chain


class GenerateTrainSets(object):
    """Train Set Generator

    Responsible for generating train sets
    to use as inputs for artificial neural networks
    and regressive models.

    """
    def __init__(self, data_set):
        """Data set inputs

        receives a data set in the form of
        an iterable matrix containing the goal
        of the model

        :param data_set: The data set as iterable
        :param goal_row: The goal for the model

        :type data_set: iterable (not a string)
        :type goal_row: int
        """
        if hasattr(data_set, '__iter__') and \
                hasattr(data_set, '__getitem__') and \
                not isinstance(data_set, str):
            self.data_set = data_set
        else:
            raise TypeError("The matrix must be an iterable"
                            "object but not a string")

    @property
    def data_set(self):
        return self.__matrix

    @data_set.setter
    def data_set(self, matrix):
        if isinstance(matrix, tuple):
            matrix = [list(i) for i in matrix]
        self.__matrix = [[float(j) if isinstance(j, int) else j for j in i]for i in matrix]

    def get_first_not_none(self, goal_row):
        """Index of first not None

        Returns the index of the first not None
        value of the goal row.

        :return: index of
        """
        goal = None
        index = 0
        while goal is None:
            goal = self.data_set[index][goal_row]
            index += 1

        return index

    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from a list

        Yield even sized chunks from a list.

        :param l: list
        :param n: number of values inside inner list
        :return: yields chunks of a list
        """
        for i in range(0, len(l), 1):
            yield l[i:i + n]

    def train_data(self, n_steps):
        data = self.chunks(self.data_set, n_steps)
        for i in data:
            if len(i) == n_steps:
                yield [f_data for f_data in chain.from_iterable(i)]



