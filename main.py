from crop_models.xlsxreader.xlsxreader import ExcelDataReader
from crop_models.dataset_preparer.data_preparer import GenerateTrainSets
data = ExcelDataReader("dados_usm_estruturados.xlsx", l1=1,
                       usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9))

# data = "dasdas"

data = GenerateTrainSets(data.data())


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


for i in data.data_set_separator(3, 9):
    print(i)