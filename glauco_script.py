from crop_models.xlsxreader.xlsxreader import ExcelDataReader
from crop_models.dataset_preparer.data_preparer import GenerateTrainSets, \
    GenerateNormalizedTrainSets
from crop_models.ann.multilayerperceptron import TimeSeriesMLPMultivariate

from sklearn.metrics import r2_score

import os
import inspect

base_path = os.path.dirname(os.path.abspath(__file__))

def r_sqrt(real, estimated):
    real_data = [i[1][0] for i in
                 real]
    predicted = [i[0] for i in estimated]

    r_q = r2_score(real_data, predicted)
    return r_q

def train_ann(goal_type='atr', n_steps=10, shape=[60, 30, 1], train_alg="train_rprop", epochs=500, goal=0.000001, adapt=False):
    base_name = "train_anns/{}_{}_steps_{}_{}_{}_{}_adapt_{}".format(goal_type, n_steps, train_alg,
                                                                     shape[0], shape[1], shape[2], adapt)
    base_path_name = "/".join([base_path, base_name])
    str_template = "{}/{}"

    if not os.path.exists(base_path_name):
        os.makedirs(base_path_name)

    goal_row = 9

    if goal_type == 'atr':
        data = ExcelDataReader("dados_usm_estruturados.xlsx", l1=1,
                              usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, goal_row))
    else:
        data = ExcelDataReader("dados_usm_estruturados.xlsx", l1=1,
                               usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10))

    gen_train_sets = GenerateNormalizedTrainSets(data.data()[0:396])
    validation_set = GenerateNormalizedTrainSets(data.data()[396:])
    dataset = gen_train_sets.normalized_data_set_separator(n_steps, goal_row, False, norm_rule="zero_one")

    mlp = TimeSeriesMLPMultivariate(shape, train_alg)

    if train_alg != "train_ncg" :
        min_error = mlp.train(dataset, save_plot=True, filename=str_template.format(base_path_name, "train_stage"),
                              epochs=epochs, goal=goal, adapt=adapt)
    else:
        min_error = mlp.train(dataset, save_plot=True, filename=str_template.format(base_path_name, "train_stage"),
                              epochs=epochs, goal=goal)


    sim = mlp.sim(x_label="{} estimado".format(goal_type.upper()), y_label="{} real".format(goal_type.upper()),
            save_plot=True, filename=str_template.format(base_path_name, "estimado_scatter"))

    mlp.out(validation_set.normalized_data_set_separator(n_steps, goal_row, False, norm_rule="zero_one"),
            x_label="{} previsto".format(goal_type.upper()), y_label="{} real".format(goal_type.upper()),
            save_plot=True, filename=str_template.format(base_path_name, "previsto_scatter"))

    predicted = mlp.out(validation_set.normalized_data_set_separator(n_steps, goal_row, False, norm_rule="zero_one"),
                x_label="{} previsto".format(goal_type.upper()), y_label="{} real".format(goal_type.upper()),
                plot_type='plot', save_plot=True, filename=str_template.format(base_path_name, "previsto_line"))

    mlp.save(str_template.format(base_path_name, "ann"))

    r_q_est = r_sqrt(gen_train_sets.normalized_data_set_separator(n_steps, goal_row, False, norm_rule="zero_one"), sim)

    r_q = r_sqrt(validation_set.normalized_data_set_separator(n_steps, goal_row, False, norm_rule="zero_one"), predicted)

    with open(str_template.format(base_path_name, "params.txt"), "wb") as f:
        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)

        for i in args:
            f.write(bytes("{}: {}\n".format(i, values[i]), encoding='utf8'))
        f.write(bytes("minimum error: {}\n".format(min_error[-1]), encoding='utf8'))

        f.write(bytes("r squared estimation: {}\n".format(r_q_est), encoding='utf8'))
        f.write(bytes("r squared forecast: {}\n".format(r_q), encoding='utf8'))

    with open(str_template.format(base_path_name, "r_squared_estimated.txt"), "wb") as f:
        f.write(bytes("{};\n".format(r_q_est), encoding='utf8'))

    with open(str_template.format(base_path_name, "r_squared_forecast.txt"), "wb") as f:
        f.write(bytes("{};\n".format(r_q), encoding='utf8'))

    return r_q

if __name__ == '__main__':
    shape = [60, 20, 1]
    for n_steps in range(1, 36):
        try:
            train_ann('atr', n_steps, shape, "train_rprop", epochs=400)
            train_ann('tch', n_steps, shape, "train_rprop", epochs=400)
            train_ann('atr', n_steps, shape, "train_ncg")
            train_ann('tch', n_steps, shape, "train_ncg")
            train_ann('atr', n_steps, shape, "train_gdx", epochs=760)
            train_ann('tch', n_steps, shape, "train_gdx", epochs=760)
        except Exception as e:
            print(e)

