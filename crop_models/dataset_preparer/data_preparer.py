from itertools import chain


class Normalizer(object):

    __norm_ranges = ("zero_one", "less_one_one")

    def __init__(self):
        pass

    def _check_normalization_rule(self, norm_range):
        if norm_range not in self.__norm_ranges:
            raise NameError("The normalization range must be one of "
                            "the options: \"zero_one\" or \"less_one_one\"")

    def normalize(self, value, min_val, max_val, norm_range="zero_one"):
        self._check_normalization_rule(norm_range)
        if norm_range == "zero_one":
            return (value - min_val)/(max_val - min_val)
        else:
            return (-2*(max_val - value/(max_val - min_val)) + 1) * -1

    def un_normalize(self, value, max_val, min_val, norm_range="zero_one"):
        self._check_normalization_rule(norm_range)
        if norm_range == "zero_one":
            return value * (max_val - min_val) + min_val
        else:
            raise ImportError("Method still not allowed")


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

        self.min_max = []

    @property
    def data_set(self):
        return self.__data_set

    @data_set.setter
    def data_set(self, data_set):
        if isinstance(data_set, tuple):
            data_set = [list(i) for i in data_set]
            data_set = [[float(j) if isinstance(j, int) else j for j in i] for i in data_set]
        self.__data_set = data_set

    @property
    def min_max(self):
        return self.__min_max

    @min_max.setter
    def min_max(self, min_max):
        for i in map(list, zip(*self.data_set)):
            no_none_i = [j for j in i if j is not None]
            min_max.append([min(no_none_i), max(no_none_i)])
        self.__min_max = min_max


    @staticmethod
    def chunks(l, n):
        """Yield successive n-sized chunks from a list

        Yield even sized chunks from a list.

        :param l: list
        :param n: number of values inside inner list

        :type l: list
        :type n: int

        :return: yields chunks of a list
        """
        for i in range(0, len(l), 1):
            yield l[i:i + n]

    def data_set_separator(self, n_steps, goal_row, goal_as_input=False):
        """Yields the train sets

        Yields the train sets with the first list being the train set by
        itself the second list the goal, the output is represented by the
        following list: [[input_0, input_1, ...input_n], [goal]]

        :param n_steps: Number of siblings periods prior goal
        :param goal_row: The row containing the model goal
        :param goal_as_input: whether or not to use the goal as
            input for the model

        :type n_steps: int
        :type goal_row: int
        :type goal_as_input: bool

        :return: Yields lists with in the form [[input_0, input_1, ...input_n], [goal]]
        :rtype: list
        """
        data = self.chunks(self.data_set, n_steps)
        for i in data:
            if len(i) == n_steps:
                transposed_data = list(map(list, zip(*i)))
                goal = transposed_data[goal_row]

                if goal_as_input:
                    train_data = transposed_data
                    if None not in goal:
                        yield [[f_data for f_data in chain.from_iterable(train_data)], [goal[-1]]]
                else:
                    train_data = transposed_data[0:goal_row]
                    if goal[-1] is not None:
                        yield [[f_data for f_data in chain.from_iterable(train_data)], [goal[-1]]]

    def update_data_set(self, data_set):
        self.__init__(data_set)


class GenerateNormalizedTrainSets(GenerateTrainSets, Normalizer):
    """Normalized Train Set Generator

    Responsible for generating normalized train sets
    to use as inputs for artificial neural networks
    and regressive models.

    """

    def __init__(self, data_set):
        GenerateTrainSets.__init__(self, data_set),
        Normalizer.__init__(self)

    def __normalize_matrix(self, data_list):
        """

        :param data_list:
        :return:
        """
        for data_l, min_max in zip(data_list, self.min_max):
            yield [self.normalize(i, min_max[0], min_max[1]) if i is not None
                   else i
                   for i in data_l]

    def normalized_data_set_separator(self, n_steps, goal_row, goal_as_input=False, norm_range="zero_one"):
        """

        :param n_steps:
        :param goal_row:
        :param goal_as_input:
        :param norm_range:
        :return:
        """
        self._check_normalization_rule(norm_range)
        data = self.chunks(self.data_set, n_steps)
        for i in data:
            if len(i) == n_steps:
                transposed_data = list(map(list, zip(*i)))
                goal = transposed_data[goal_row][-1]
                if goal_as_input:
                    train_data = self.__normalize_matrix(transposed_data)
                    if None not in transposed_data[goal_row]:
                        yield [[f_data for f_data in chain.from_iterable(train_data)],
                               [self.normalize(goal,
                                self.min_max[goal_row][0],
                                self.min_max[goal_row][1])]
                               ]
                else:
                    goal = transposed_data[goal_row][-1]
                    train_data = self.__normalize_matrix(transposed_data[0:goal_row])
                    if goal is not None:
                        yield [[f_data for f_data in chain.from_iterable(train_data)],
                               [self.normalize(goal,
                                self.min_max[goal_row][0],
                                self.min_max[goal_row][1])]]

