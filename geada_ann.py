from crop_models.ann.multilayerperceptron import *

import matplotlib  as pl
from crop_models.xlsxreader import ExcelDataReader
from crop_models.dataset_preparer.data_preparer import GenerateTrainSets, \
    GenerateNormalizedTrainSets

import numpy as np
import pylab as pl
import neurolab as nl

topolgia = [20, 1]
step = 1
momentum = 0.9
val_size = 1
epochs = 300

data = ExcelDataReader("parametros_geada_cafe.xlsx", l1=1, usecols=(0, 1))

data = data.data()

update_data = []
sepparated_data = []
data_until_none = [[float(data[0][0].split(",")[1])],[data[0][1]]]

# Sort data and get the 3 lowest temperatures
counter = 1

for i in data[1:]:
    if i[1] is not None:
        update_data.append([sorted(data_until_none[0])[0:3], data_until_none[1]])
        data_until_none = [[], []]
        data_until_none[1].append(i[1])
        data_until_none[0].append(float(i[0].split(",")[1]))
    else:
        data_until_none[0].append(float(i[0].split(",")[1]))
    counter += 1

    if counter==len(data):
        update_data.append([sorted(data_until_none[0])[0:3], data_until_none[1]])


for i in update_data:
    print(i)

# train_data = []
# for i in update_data:
#     flat_list = []
#     flat_list.extend(i[0])
#     flat_list.append(i[1])
#     train_data.append(flat_list)
#
# gen_train_set = GenerateNormalizedTrainSets(train_data)



mlp = TimeSeriesMLPMultivariate([20, 1], train_alg="train_rprop")
min_error = mlp.train(update_data, save_plot=True, epochs=250, goal=0.000001)
mlp.sim(x_label="Taxa de estimada", y_label="Taxa de perda real", save_plot=True, filename="estimado_scatter")
mlp.sim(x_label="evento", y_label="Taxa de perda real", save_plot=True, plot_type='plot', filename="estimado_line")

print(mlp.ann.__dict__)

print("entrada a camada escondida\n", mlp.ann.layers[0].__dict__)
print("camada escondida a camada de saida\n", mlp.ann.layers[1].__dict__)
# min_error = mlp.train(dataset, save_plot=True, epochs=470, goal=0.000001)


# train_data_multi = {"data": train_data,
#                     "periodo": 1,
#                     "val_size": val_size,
#                     "hiddenlayers": topolgia,
#                     "step": 0,
#                     "goal_row": 3}
#
# multi_variate = TemporalMultivariateTrainAlg(**train_data_multi)
# print(multi_variate._train_data(False))

# multi_variate_error = multi_variate.error(momentum, epochs, adapt=False, goal=0.001)
# error = multi_variate_error
#
# numero_de_epocas = len(multi_variate_error)
#
# pl.plot(multi_variate_error)
# pl.xlabel('Epoch number(Numero da epoca)')
# pl.ylabel('erro (Default SSE)')
#
# pl.show()
#
# pl.plot(multi_variate._target(), color='red')
# pl.plot(multi_variate._target(), color='red')
# pl.plot(multi_variate.out(), color='blue')
# pl.ylabel('Previsao(Padronizada)')
# pl.xlabel('Epoch number(Numero da epoca)')
# pl.show()
#
total = 0
for i, j in zip(mlp._target_data, mlp.sim(plot=False, save_plot=False)):
    total+=abs((j[0] - i[0]))
    print("Real: ", j[0], "Previsto: ", i[0], "Diferenca: ", (j[0] - i[0]))

print((total/13) * 100)
