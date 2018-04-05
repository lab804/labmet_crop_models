from itertools import chain
from collections import defaultdict


class Normalizer(object):
    """Normalizer

    Container of normalizer functions intended for 
    statistical analysis and modeling.

    """
    __norm_rule = ("zero_one", "less_one_one")
    __norm_range = ([0, 1], [-1, 1])

    def _check_normalization_rule(self, norm_rule):
        """Check normalization rule

        Checks if normalization rule is available.

        :param norm_rule: Normalization rule
        :return: None, raises if not implemented
        """
        if norm_rule not in self.__norm_rule:
            raise NameError("The normalization range must be one of "
                            "the options: \"zero_one\" or \"less_one_one\"")

    def print_normalization_rule(self):
        """Print normalization rules

        Prints all available rules and ranges.

        :return: None, prints available normalization rule.
        """
        for i in zip(self.__norm_rule, self.__norm_range):
            print("{}, [{}, {}]".format(i[0], i[1][0], i[1][1]))

    def normalize(self, value, min_val, max_val, norm_rule="zero_one"):
        """Normalize

        Mathematical implementations of normalizations equations.

        :param value: Value to be normalized
        :param min_val: Minimum value from the domain of the
         value to be normalized
        :param max_val: Maximum value from the domain of the
         value to be normalized
        :param norm_rule: Normalization rule to be used

        :type value: float
        :type min_val: float
        :type max_val: float
        :type norm_rule: str


        :return: Normalized value inside value domain.
        :rtype: float.
        """
        self._check_normalization_rule(norm_rule)
        if norm_rule == "zero_one":
            return (value - min_val)/(max_val - min_val)
        else:
            return ((-2*(max_val - value)/(max_val - min_val)) + 1) * -1

    def un_normalize(self, value, min_val, max_val, norm_rule="zero_one"):
        """Un normalize

        Inverse mathematical implementations of the normalizations
        equations, intended to return the value to its original domain.

        :param value: Value to be un-normalized
        :param min_val: Minimum value from the domain of the
         value to be un-normalized
        :param max_val: Maximum value from the domain of the
         value to be un-normalized
        :param norm_rule: Normalization rule that the value
         was originally normalized.

        :type value: float
        :type min_val: float
        :type max_val: float
        :type norm_rule: str


        :return: Value in its original domain.
        :rtype: float.
        """
        self._check_normalization_rule(norm_rule)
        if norm_rule == "zero_one":
            return value * (max_val - min_val) + min_val
        else:
            raise Exception("Method still not allowed")


class GenerateTrainSets(object):
    """Train Set Generator

    Responsible for generating train sets
    to use as inputs for artificial neural networks
    and regressive models.

    """
    def __init__(self, data_set, min_max=None):
        """Dataset inputs

        receives a dataset in the form of
        an iterable matrix containing the goal
        of the model

        :param data_set: The dataset as iterable
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

        if min_max is None:
            self.min_max = self.__set_min_max()
        else:
            self.min_max = min_max

    @property
    def data_set(self):
        return self.__data_set

    @data_set.setter
    def data_set(self, data_set):
        if isinstance(data_set, tuple):
            data_set = [list(i) for i in data_set]
            data_set = [[float(j) if isinstance(j, int) else j for j in i] for i in data_set]
        self.__data_set = data_set


    def __set_min_max(self):
        min_max = []
        for i in map(list, zip(*self.data_set)):
            no_none_i = [j for j in i if j is not None]
            max_val = max(no_none_i)
            min_val = min(no_none_i)
            min_max.append([min_val, max_val])
        return min_max


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

    def data_set_separator(self, n_steps, goal_row, delay=1, goal_as_input=False):
        """Yields the train sets

        Yields the train sets with the first list being the train set by
        itself the second list the goal, the output is represented by the
        following list: [[input_0, input_1, ...input_n], [goal]].

        :param n_steps: Number of siblings periods prior goal
        :param goal_row: The row containing the model goal
        :param goal_as_input: whether or not to use the goal as
            input for the model

        :type n_steps: int
        :type goal_row: int
        :type goal_as_input: bool

        :return: Yields lists with the
         form [[input_0, input_1, ...input_n], [goal]]
        :rtype: list
        """


        data = [i for i in self.chunks(self.data_set, n_steps)]
        train_set = data[0: -delay]
        goal_set = [list(map(list, zip(*i)))[goal_row][-1] for i in data[delay:]]

        for i, j in zip(train_set, goal_set):
            if len(i) == n_steps:
                transposed_data = list(map(list, zip(*i)))
                goal =  j
                if goal_as_input:
                    train_data = transposed_data
                    if goal is not None:
                        yield [[f_data for f_data in chain.from_iterable(train_data)], [goal[-1]]]
                else:
                    train_data = transposed_data[0:goal_row]
                    data = [f_data for f_data in chain.from_iterable(train_data)]
                    if goal is not None and None not in data:
                        yield [data, [goal]]

    def update_data_set(self, data_set):
        """Updates instance with new dataset

        Updates an instanced object with a new dataset to be processed

        :param data_set:
        :return:
        """
        self.__init__(data_set)


class GenerateNormalizedTrainSets(GenerateTrainSets, Normalizer):
    """Normalized Train Set Generator

    Responsible for generating normalized train sets
    to use as inputs for artificial neural networks
    and regressive models.

    """

    def __init__(self, data_set, min_max=None):
        GenerateTrainSets.__init__(self, data_set, min_max)
        Normalizer.__init__(self)

    def normalized_min_max(self, norm_rule="zero_one"):
        return [[self.normalize(i[0], *i, norm_rule=norm_rule),
                 self.normalize(i[1], *i, norm_rule=norm_rule)] for i in self.min_max]

    def _normalize_matrix(self, data_list, norm_rule="zero_one"):
        """Normalize matrices

        Helper function to normalize multidimensional lists.

        :param data_list:
        :return: yield list with normalized values
        :rtype: generator
        """
        for data_l, min_max in zip(data_list, self.min_max):
            try:
                yield [self.normalize(i, min(min_max), max(min_max), norm_rule) if i is not None
                       else i
                       for i in data_l]

            except ZeroDivisionError as e:
                print(min_max[0], print(min_max[1]))
                print(e)

    def _normalize_and_concatenate_dataset(self, input_data, norm_rule, goal_row, goal_as_input):

        if goal_as_input:
            train_data = self._normalize_matrix(input_data, norm_rule)
        else:
            train_data = self._normalize_matrix(input_data[0:goal_row], norm_rule)

        data = [f_data for f_data in chain.from_iterable(train_data)]

        return data

    def normalized_data_set_separator(self, n_steps, goal_row, goal_as_input=False, norm_rule="zero_one", delay=1):
        """Creates normalized datasets

        Creates normalized datasets intended to be used in
        the training stage of regressive models, specially
        neural networks. This methods returns an generator
        of lists with normalized values with the following
        form: [[input_0, input_1, ...input_n], [goal]].

        :param n_steps: Number of accumulated convoluted steps
         to be returned in the first index of the generated lists.
        :param goal_row: The row containing the model goal
        :param goal_as_input: whether or not to use the goal as
            input for the model

        :type n_steps: int
        :type goal_row: int
        :type goal_as_input: bool

        :return: Yields lists with normalized values
         in the form [[input_0, input_1, ...input_n], [goal]]
        :rtype: generator
        """



        self._check_normalization_rule(norm_rule)

        data = list(self.chunks(self.data_set, n_steps))
        train_set = data[0: -delay]
        goal_set = [list(map(list, zip(*i)))[goal_row][-1] for i in data[delay:]]

        # data = self.chunks(self.data_set, n_steps)

        for i, j in zip(train_set, goal_set):
            if len(i) == n_steps:
                transposed_data = list(map(list, zip(*i)))
                goal = j

                data = self._normalize_and_concatenate_dataset(transposed_data, norm_rule, goal_row, goal_as_input)

                if goal is not None and None not in data:
                    yield [data,
                               [self.normalize(goal,
                                self.min_max[goal_row][0],
                                self.min_max[goal_row][1])]]


    def un_normalize_list(self, uni_dimensional_list, sibling_row, norm_rule="zero_one"):
        """Un-normalize uni-dimensional lists

        Un-normalizes one dimension lists, returning
        it to its original form and domain.

        :param uni_dimensional_list: One dimension list
        :param sibling_row: The row of the original dataset that
         this list represents.
        :param norm_rule: Normalization rule

        :type uni_dimensional_list: list or generator
        :type sibling_row: int
        :type norm_rule: str

        :return: List with its original form and domain
        :rtype: list
        """
        return [self.un_normalize(i,
                                  self.min_max[sibling_row][0],
                                  self.min_max[sibling_row][1],
                                  norm_rule)
                for i in uni_dimensional_list]

class GenerateSeasonedNormalizedTrainSets(GenerateNormalizedTrainSets):
    """Normalized Train Set Generator

    Responsible for generating normalized train sets
    to use as inputs for artificial neural networks
    and regressive models.

    """


    def _separate_in_seasons(self, data, register_per_season, n_seasons):
        count_seasons = 0
        count_register = 0
        new_data = []
        for i in data:
            new_data.append([count_seasons, i])
            count_register += 1

            if count_register == register_per_season:
                count_seasons += 1
                count_register = 0
            if count_seasons == n_seasons:
                count_seasons = 0
        return new_data

    def data_set_separator(self, n_steps, goal_row, n_seasons, register_per_season, goal_as_input=False, delay=1):
        """Yields the train sets

        Yields the train sets with the first list being the train set by
        itself the second list the goal, the output is represented by the
        following list: [[input_0, input_1, ...input_n], [goal]].

        :param n_steps: Number of siblings periods prior goal
        :param goal_row: The row containing the model goal
        :param goal_as_input: whether or not to use the goal as
            input for the model

        :type n_steps: int
        :type goal_row: int
        :type goal_as_input: bool

        :return: Yields lists with the
         form [[input_0, input_1, ...input_n], [goal]]
        :rtype: list
        """
        data = list(self.chunks(self.data_set, n_steps))
        train_set = self._separate_in_seasons(data, register_per_season, n_seasons)[0: - delay]
        goal_set = [list(map(list, zip(*i)))[goal_row][-1] for i in data[delay:]]

        return_data = defaultdict(list)

        for i, j in zip(train_set, goal_set):
            if len(i[1]) == n_steps:
                register_domain = i[0]
                transposed_data = list(map(list, zip(*i[1])))
                goal = j

                if goal_as_input:
                    train_data = transposed_data

                else:
                    train_data = transposed_data[0:goal_row]

                data = [f_data for f_data in chain.from_iterable(train_data)]
                # data = self._normalize_and_concatenate_dataset(transposed_data, norm_rule, goal_row, goal_as_input)

                if goal is not None and None not in data:
                    ann_data = ([data,
                                 [goal]])
                    return_data[str(register_domain)].append(ann_data)

        return return_data

    def normalized_data_set_separator(self, n_steps, goal_row, n_seasons, register_per_season, goal_as_input=False, norm_rule="zero_one", delay=1):
        """Creates normalized datasets

        Creates normalized datasets intended to be used in
        the training stage of regressive models, specially
        neural networks. This methods returns an generator
        of lists with normalized values with the following
        form: [[input_0, input_1, ...input_n], [goal]].

        :param n_steps: Number o1f accumulated convoluted steps
         to be returned in the first index of the generated lists.
        :param goal_row: The row containing the model goal
        :param goal_as_input: whether or not to use the goal as
            input for the model

        :type n_steps: int
        :type goal_row: int
        :type goal_as_input: bool

        :return: Yields lists with normalized values
         in the form [[input_0, input_1, ...input_n], [goal]]
        :rtype: generator
        """
        self._check_normalization_rule(norm_rule)
        data = list(self.chunks(self.data_set, n_steps))
        train_set = self._separate_in_seasons(data, register_per_season, n_seasons)[0: - delay]
        goal_set = [list(map(list, zip(*i)))[goal_row][-1] for i in data[delay:]]


        return_data = defaultdict(list)

        for i, j in zip(train_set, goal_set):
            if len(i[1]) == n_steps:
                register_domain = i[0]
                transposed_data = list(map(list, zip(*i[1])))
                goal = j

                data = self._normalize_and_concatenate_dataset(transposed_data, norm_rule, goal_row, goal_as_input)

                if goal is not None and None not in data:
                    ann_data = ([data,
                                 [self.normalize(goal,
                                                 self.min_max[goal_row][0],
                                                 self.min_max[goal_row][1])]])
                    return_data[str(register_domain)].append(ann_data)

        return return_data