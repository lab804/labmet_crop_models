from itertools import chain


class GenerateTrainSets(object):
    """

    """
    def __init__(self, data_set, goal_row):
        if hasattr(data_set, '__iter__'):
            self.data_set = data_set
        else:
            raise TypeError("The matrix must be an iterable"
                            "object")
        self.goal_row = goal_row

    @property
    def data_set(self):
        return self.__matrix

    @data_set.setter
    def data_set(self, matrix):

        if isinstance(matrix, tuple):
            matrix = [list(i) for i in matrix]
        self.__matrix = [[float(j) if isinstance(j, int) else j for j in i]for i in matrix]

    def get_first_none(self):
        goal = None
        index = 0
        while goal is None:
            goal = self.data_set[index][-1]
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



