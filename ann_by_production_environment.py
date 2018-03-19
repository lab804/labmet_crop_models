from crop_models.xlsxreader.xlsxreader import ExcelDataReader
from crop_models.dataset_preparer.data_preparer import GenerateTrainSets, \
    GenerateNormalizedTrainSets, GenerateSeasonedNormalizedTrainSets
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


def  open_dataset(start=0, stop=396, n_steps=10):

    goal_row = 10

    data_amb_a = ExcelDataReader("dados_uiracemapolis_estruturados.xlsx", l1=1,
                           usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12))
    data_amb_c = ExcelDataReader("dados_uiracemapolis_estruturados.xlsx", l1=1,
                           usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14))
    data_amb_d = ExcelDataReader("dados_uiracemapolis_estruturados.xlsx", l1=1,
                                 usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16))
    data_ambs = [data_amb_a, data_amb_c, data_amb_d]

    gen_train_sets = [GenerateSeasonedNormalizedTrainSets(i.data()[start:stop]) for i in data_ambs]

    datasets = [i.normalized_data_set_separator(n_steps, goal_row, 12, 3, False, norm_rule="less_one_one") for i in gen_train_sets]
    dic_default = datasets[0]
    for i in datasets[1:]:
        for k, v in i.items():
            dic_default[k].append(v)
    return dic_default

def month_ann(goal_type='atr', n_steps=10,
              shape=[60, 1], train_alg="train_rprop",
              epochs=500, goal=0.000001, adapt=False, show=1):
    base_name = "train_anns/{}_{}_steps_{}_{}_adapt_{}".format(goal_type, n_steps, train_alg,
                                                                     "_".join(map(str, shape)), adapt)
    base_path_name = "/".join([base_path, base_name])
    str_template = "{}/{}_{}_mes"

    if not os.path.exists(base_path_name):
        os.makedirs(base_path_name)

    goal_row = 9

    gen_train_sets = open_dataset(n_steps=n_steps, start=0, stop=396)
    validation_set = open_dataset(n_steps=n_steps, start=396, stop=-1)

    #todo finish this

    dataset = gen_train_sets.normalized_data_set_separator(n_steps, goal_row, 12, 3, False, norm_rule="less_one_one")
    for k, v in dataset.items():
        mlp = TimeSeriesMLPMultivariate(shape, train_alg)
        if train_alg != "train_ncg":
            min_error = mlp.train(v, save_plot=True, filename=str_template.format(base_path_name, "train_stage", k),
                                  epochs=epochs, goal=goal, adapt=adapt, show=show)
        else:
            min_error = mlp.train(v, save_plot=True, filename=str_template.format(base_path_name, "train_stage", k),
                                  epochs=epochs, goal=goal)

        sim = mlp.sim(x_label="{} estimado".format(goal_type.upper()), y_label="{} real".format(goal_type.upper()),
                      save_plot=True, filename=str_template.format(base_path_name, "estimado_scatter", k))

        mlp.out(validation_set.normalized_data_set_separator(n_steps, goal_row, 12, 3, False, norm_rule="less_one_one")[k],
                x_label="{} previsto".format(goal_type.upper()), y_label="{} real".format(goal_type.upper()),
                save_plot=True, filename=str_template.format(base_path_name, "previsto_scatter", k))

        val_data_set = validation_set.normalized_data_set_separator(n_steps, goal_row, 12, 3, False, norm_rule="less_one_one")[k]
        predicted = mlp.out(
            val_data_set,
            x_label="{} previsto".format(goal_type.upper()), y_label="{} real".format(goal_type.upper()),
            plot_type='plot', save_plot=True, filename=str_template.format(base_path_name, "previsto_line", k))

        mlp.save(str_template.format(base_path_name, "ann", k))
        try:
            r_q_est = r_sqrt(gen_train_sets.normalized_data_set_separator(n_steps, goal_row, 12, 3, False, norm_rule="less_one_one")[k],
                             sim)

            r_q = r_sqrt(validation_set.normalized_data_set_separator(n_steps, goal_row, 12, 3, False, norm_rule="less_one_one")[k],
                         predicted)
        except:
            r_q_est = 0
            r_q = 0

        with open("{}/{}".format(base_path_name, "params_{}_mes.txt".format(k)), "wb") as f:
            frame = inspect.currentframe()
            args, _, _, values = inspect.getargvalues(frame)

            for i in args:
                f.write(bytes("{}: {}\n".format(i, values[i]), encoding='utf8'))
            f.write(bytes("minimum error: {}\n".format(min_error[-1]), encoding='utf8'))

            f.write(bytes("r squared estimation: {}\n".format(r_q_est), encoding='utf8'))
            f.write(bytes("r squared forecast: {}\n".format(r_q), encoding='utf8'))

        with open("{}/{}".format(base_path_name, "r_squared_estimated{}_mes.txt".format(k)), "wb") as f:
            f.write(bytes("{};\n".format(r_q_est), encoding='utf8'))

        with open("{}/{}".format(base_path_name, "r_squared_forecast{}_mes.txt".format(k)), "wb") as f:
            f.write(bytes("{};\n".format(r_q), encoding='utf8'))



if __name__ == '__main__':

    # # month_ann('atr', 3, [60, 20, 1], "train_rprop", epochs=400)
    shape = [40, 10,1]
    for n_steps in range(15, 36):
        # try:
            # month_ann('atr', n_steps, shape, "train_rprop", epochs=400)

            month_ann('tch', n_steps, shape, "train_rprop", epochs=600, show=50)
            # month_ann('tch', n_steps, shape, "train_rprop", epochs=600, show=50, adapt=True)

            # month_ann('atr', n_steps, shape, "train_ncg")
            month_ann('tch', n_steps, shape, "train_ncg")
            # month_ann('atr', n_steps, shape, "train_gdx", epochs=760)
            month_ann('tch', n_steps, shape, "train_gdx", epochs=900, show=50)
            # month_ann('tch', n_steps, shape, "train_gdx", adapt=True, epochs=900, show=50)
        # except Exception as e:
        #     print(e)
