from crop_models.xlsxreader.xlsxreader import ExcelDataReader
from crop_models.dataset_preparer.data_preparer import GenerateTrainSets, \
    GenerateNormalizedTrainSets
from crop_models.ann.multilayerperceptron import TimeSeriesMLPMultivariate

data = ExcelDataReader("dados_usm_estruturados.xlsx", l1=1,
                       usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))

# data = "dasdas"

gen_train_sets = GenerateTrainSets(data.data())


# numero_de_valores = 0
# for i in gen_train_sets.data_set_separator(18, 9, goal_as_input=False):
#     numero_de_valores += 1
#     print(i)


gen_norm_train_set = GenerateNormalizedTrainSets(data.data())
dataset = gen_norm_train_set.normalized_data_set_separator(1, 9, True, norm_rule="zero_one")

# print(gen_norm_train_set.normalized_min_max())

mlp = TimeSeriesMLPMultivariate([4, 4, 1])

min_error = mlp.train(dataset)
print(min_error[-1])

# data_test = gen_norm_train_set.data_set
# matrix = list(map(list, zip(*data_test)))[0]
# print(matrix)
# norm_data_test = gen_norm_train_set.normalized_data_set_separator(1, 9, goal_as_input=True)
# norm_data_test = [i[0][0] for i in norm_data_test]
# normalizado = gen_norm_train_set.un_normalize_list(norm_data_test, 0)
# for i, j in zip(normalizado, norm_data_test):
#     print("real", i, "padronizado: ", j)
# gen_norm_train_set.print_normalization_rule()

    # print(gen_norm_train_set.data_set_separator())