from crop_models.xlsxreader.xlsxreader import ExcelDataReader
from crop_models.dataset_preparer.data_preparer import GenerateTrainSets, \
    GenerateNormalizedTrainSets, GenerateSeasonedNormalizedTrainSets
from crop_models.ann.multilayerperceptron import TimeSeriesMLPMultivariate

from sklearn.metrics import r2_score

from collections import defaultdict


import os
import inspect

base_path = os.path.dirname(os.path.abspath(__file__))

def r_sqrt(real, estimated):
    real_data = [i[1][0] for i in
                 real]
    predicted = [i[0] for i in estimated]

    r_q = r2_score(real_data, predicted)
    return r_q


def  open_dataset(start=0, stop=396, n_steps=10, n_of_seasons=12, periods_by_season=3, validation=False):

    goal_row = 9

    data_amb_a = ExcelDataReader("dados_uiracemapolis_estruturados.xlsx", l1=1,
                           usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 12))
    data_amb_c = ExcelDataReader("dados_uiracemapolis_estruturados.xlsx", l1=1,
                           usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 14))
    data_amb_d = ExcelDataReader("dados_uiracemapolis_estruturados.xlsx", l1=1,
                                 usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 16))

    data_ambs = [data_amb_a, data_amb_c, data_amb_d]

    if not validation:
        gen_train_sets = [GenerateSeasonedNormalizedTrainSets(i.data()[start:stop]) for i in data_ambs]
    else:
        min_max = GenerateNormalizedTrainSets(data_amb_a.data()[0:start]).min_max
        # for i in GenerateNormalizedTrainSets(data_amb_a.data()[0:start]).data_set_separator(n_steps, goal_row):
        #     print(i)
        gen_train_sets = [GenerateSeasonedNormalizedTrainSets(i.data()[start:stop], min_max) for i in data_ambs]

    # for i in gen_train_sets[0].data_set_separator(n_steps, 9, False):
    #     print(i)


    # datasets = [i.normalized_data_set_separator(n_steps, goal_row, n_of_seasons, periods_by_season,
    #                                             goal_as_input=False,
    #                                             norm_rule="less_one_one") for i in gen_train_sets]
    datasets = [i.data_set_separator(n_steps, goal_row, n_of_seasons, periods_by_season,
                                                goal_as_input=False,
                                                ) for i in gen_train_sets]
    dic_default = defaultdict(list)
    count=-1
    for i in datasets:
        for k, v in i.items():
            for j in v:
                j[0].insert(-1, count)
            dic_default[k].extend(v)
        count+=1

    return dic_default

def month_ann(goal_type='tch', n_steps=10,
              shape=[60, 1], train_alg="train_rprop",
              epochs=500, goal=0.1, adapt=False, show=1):
    base_name = "train_anns/{}_{}_steps_{}_{}_adapt_{}".format(goal_type, n_steps, train_alg,
                                                                     "_".join(map(str, shape)), adapt)
    base_path_name = "/".join([base_path, base_name])
    str_template = "{}/{}_{}_mes"

    if not os.path.exists(base_path_name):
        os.makedirs(base_path_name)

    # goal_row = 9

    gen_train_sets = open_dataset(n_steps=n_steps, start=0, stop=396)
    validation_set = open_dataset(n_steps=n_steps, start=396 - n_steps, stop=-1, validation=True)

    # dataset = gen_train_sets.normalized_data_set_separator(n_steps, goal_row, 12, 3, False, norm_rule="less_one_one")
    for i in gen_train_sets.keys():
        print(len(gen_train_sets[i]))
        # print(len(validation_set[i]))

    for k, v in gen_train_sets.items():
        if len(validation_set[k]) > 0:
            mlp = TimeSeriesMLPMultivariate(shape, train_alg, error_function='sse')
            if train_alg != "train_ncg" and train_alg != "train_cg":

                tries = 0
                print(v)
                while tries < 5:
                    min_error = mlp.train(v, save_plot=True, filename=str_template.format(base_path_name, "train_stage", k),
                                          epochs=epochs, goal=goal, adapt=adapt, show=show)
                    if min_error[-1] < 0.7:
                        tries = 5

                    tries+=1
            else:

                min_error = mlp.train(v, save_plot=True, filename=str_template.format(base_path_name, "train_stage", k),
                                          epochs=epochs, goal=goal)

            sim = mlp.sim(x_label="{} estimado".format(goal_type.upper()), y_label="{} real".format(goal_type.upper()),
                          save_plot=True, filename=str_template.format(base_path_name, "estimado_scatter", k))

            mlp.out(validation_set[k],
                    x_label="{} previsto".format(goal_type.upper()), y_label="{} real".format(goal_type.upper()),
                    save_plot=True, filename=str_template.format(base_path_name, "previsto_scatter", k))

            predicted = mlp.out(
                validation_set[k],
                x_label="{} previsto".format(goal_type.upper()), y_label="{} real".format(goal_type.upper()),
                plot_type='plot', save_plot=True, filename=str_template.format(base_path_name, "previsto_line", k))

            mlp.save(str_template.format(base_path_name, "ann", k))
            try:
                r_q_est = r_sqrt(v,
                                 sim)

                r_q = r_sqrt(v,
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
    # open_dataset(start=396, stop=-1, n_steps=1, n_of_seasons=12, periods_by_season=3, validation=True)

    month_ann('TCH', 5, [10, 5, 1], "train_gdx", epochs=1500, goal=0.005)
    # shape = [40, 10,1]
    # for n_steps in range(1, 36):
    #     # try:
    #         # month_ann('atr', n_steps, shape, "train_rprop", epochs=400)
    #
    #         month_ann('tch', n_steps, shape, "train_rprop", epochs=600, show=50)
    #         # month_ann('tch', n_steps, shape, "train_rprop", epochs=600, show=50, adapt=True)
    #
    #         # month_ann('atr', n_steps, shape, "train_ncg")
    #         month_ann('tch', n_steps, shape, "train_cg", show=50)
    #         # month_ann('atr', n_steps, shape, "train_gdx", epochs=760)
    #         month_ann('tch', n_steps, shape, "train_gdx", epochs=1200, show=50)
    #         # month_ann('tch', n_steps, shape, "train_gdx", adapt=True, epochs=900, show=50)
    #     # except Exception as e:
    #     #     print(e)
