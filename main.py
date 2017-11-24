from crop_models.xlsxreader.xlsxreader import ExcelDataReader
from crop_models.dataset_preparer.data_preparer import GenerateTrainSets, \
    GenerateNormalizedTrainSets
data = ExcelDataReader("dados_usm_estruturados.xlsx", l1=1,
                       usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))

# data = "dasdas"

gen_train_sets = GenerateTrainSets(data.data())


def get_first_none(matrix):
    goal = None
    index = 0
    while goal is None:
        goal = matrix[index][-1]
        index += 1

    return index


def chunks(l, n):
    for i in range(0, len(l), 1):
        yield l[i:i+n]


def gen_train_set(matrix, amounth_before):
    max_befor = get_first_none(matrix)
    for i in chunks(matrix, amounth_before):
       print(i)

# numero_de_valores = 0
# for i in gen_train_sets.data_set_separator(18, 9, goal_as_input=False):
#     numero_de_valores += 1
#     print(i)


gen_norm_train_set = GenerateNormalizedTrainSets(data.data())

count = 0
print(gen_norm_train_set.min_max)
for i, j in zip(gen_norm_train_set.normalized_data_set_separator(1, 9, True, norm_rule="zero_one"),
             gen_norm_train_set.data_set_separator(1, 9, goal_as_input=True)):
    count += 1
    print(count)
    print(j)
    print(i)

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