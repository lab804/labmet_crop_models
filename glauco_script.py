from crop_models.xlsxreader.xlsxreader import ExcelDataReader
from crop_models.dataset_preparer.data_preparer import GenerateTrainSets, \
    GenerateNormalizedTrainSets, GenerateSeasonedNormalizedTrainSets
from crop_models.ann.multilayerperceptron import TimeSeriesMLPMultivariate
from functools import reduce
from sklearn.metrics import r2_score

import os
import inspect
from collections import defaultdict

base_path = os.path.dirname(os.path.abspath(__file__))

def r_sqrt(real, estimated):
    real_data = [i[1][0] for i in
                 real]
    predicted = [i[0] for i in estimated]

    r_q = r2_score(real_data, predicted)
    return r_q

def train_ann(goal_type='atr', n_steps=10,
              shape=[60, 30, 1], train_alg="train_rprop",
              epochs=500, goal=0.000001, adapt=False):

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

def generate_sum(data_ambs, goal_row, sum_steps):
    new_amb = list(map(list, zip(*data_ambs)))
    avg_data = new_amb[1:-1]
    rainfall = new_amb[0]
    prod = new_amb[goal_row]

    data_amb = []

    rain_pre_append = []

    for i in chunks(rainfall, sum_steps):
        if None not in i:
            sum_rain = sum(i)
            rain_pre_append.append(sum_rain)
        else:
            rain_pre_append.append(None)
    data_amb.append(rain_pre_append)

    for avg in avg_data:
        pre_append = []
        for i in chunks(avg, sum_steps):
            if None not in i:
                average = reduce(lambda x, y: x + y, i) / float(len(i))

                pre_append.append(average)

            else:
                pre_append.append(None)
        data_amb.append(pre_append)

    data_amb_a = list(map(list, zip(*data_amb)))
    for i, j in zip(data_amb_a, prod):
        i.append(j)
    return (data_amb_a)

def open_dataset(start=0, stop=396, n_steps=10, delay=1, n_of_seasons=12, periods_by_season=3, validation=False, sum_steps=5):
    goal_row = 9

    data_amb_a = ExcelDataReader("dados_uiracemapolis_estruturados.xlsx", l1=1,
                                 usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10))
    data_amb_c = ExcelDataReader("dados_usm_estruturados.xlsx", l1=1,
                                 usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10))

    data_ambs = [generate_sum(data_amb_a.data(), 9, sum_steps), generate_sum(data_amb_c.data(), 9, sum_steps)]

    # print(data_ambs[0])
    if not validation:
        gen_train_sets = [GenerateSeasonedNormalizedTrainSets(i[start:stop]) for i in data_ambs]
    else:
        min_max_base = generate_sum(data_amb_a.data(), 9, sum_steps)
        min_max_base.extend(generate_sum(data_amb_c.data(), 9, sum_steps))
        min_max = GenerateNormalizedTrainSets(min_max_base[0:start]).min_max
        gen_train_sets = [GenerateSeasonedNormalizedTrainSets(i[start:stop], min_max) for i in data_ambs]

    datasets = [i.normalized_data_set_separator(n_steps, goal_row, n_of_seasons, periods_by_season,
                                                goal_as_input=False,
                                                norm_rule="less_one_one",
                                                delay=delay) for i in gen_train_sets]
    dic_default = defaultdict(list)
    count = -1
    for i in datasets:
        for k, v in i.items():
            dic_default[k].extend(v)
        count += 1

    return dic_default


def month_ann(goal_type='atr', n_steps=10, delay=1,
              shape=[60, 1], train_alg="train_rprop",
              epochs=500, goal=0.005, adapt=False, show=1,
              sum_steps=5):
    base_name = "train_anns_summed_6/{}_{}_steps_{}_delay_{}_{}_adapt_{}".format(goal_type, n_steps, delay, train_alg,
                                                                     "_".join(map(str, shape)), adapt)
    base_path_name = "/".join([base_path, base_name])
    str_template = "{}/{}_{}_mes"

    if not os.path.exists(base_path_name):
        os.makedirs(base_path_name)

    # goal_row = 9

    # if goal_type == 'atr':
    #     data = ExcelDataReader("dados_usm_estruturados.xlsx", l1=1,
    #                            usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, goal_row))
    # else:
    #     data = ExcelDataReader("dados_usm_estruturados.xlsx", l1=1,
    #                            usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10))


    gen_train_sets = open_dataset(n_steps=n_steps, delay=delay, start=0, stop=396, sum_steps=sum_steps)

    validation_set = open_dataset(n_steps=n_steps, delay=delay, start=396 - n_steps, stop=-1,
                                  validation=True, sum_steps=sum_steps)

    for k, v in gen_train_sets.items():
        if len(validation_set[k]) > 0:
            mlp = TimeSeriesMLPMultivariate(shape, train_alg, error_function='sse')
            if train_alg != "train_ncg" and train_alg != "train_cg":

                tries = 0
                while tries < 5:
                    min_error = mlp.train(v, save_plot=True, filename=str_template.format(base_path_name, "train_stage", k),
                                          epochs=epochs, goal=goal, adapt=adapt, show=show)
                    if min_error[-1] < 0.12:
                        tries = 5

                    tries+=1
            else:

                min_error = mlp.train(v, save_plot=True, filename=str_template.format(base_path_name, "train_stage", k),
                                          epochs=epochs, goal=goal, rr=0.01)

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

    # for k, v in dataset.items():
    #     mlp = TimeSeriesMLPMultivariate(shape, train_alg)
    #     if train_alg != "train_ncg":
    #         min_error = mlp.train(v, save_plot=True, filename=str_template.format(base_path_name, "train_stage", k),
    #                               epochs=epochs, goal=goal, adapt=adapt, show=show)
    #     else:
    #         min_error = mlp.train(v, save_plot=True, filename=str_template.format(base_path_name, "train_stage", k),
    #                               epochs=epochs, goal=goal)
    #
    #     sim = mlp.sim(x_label="{} estimado".format(goal_type.upper()), y_label="{} real".format(goal_type.upper()),
    #                   save_plot=True, filename=str_template.format(base_path_name, "estimado_scatter", k))
    #
    #     mlp.out(validation_set.normalized_data_set_separator(n_steps, goal_row, 12, 3, False, norm_rule="less_one_one")[k],
    #             x_label="{} previsto".format(goal_type.upper()), y_label="{} real".format(goal_type.upper()),
    #             save_plot=True, filename=str_template.format(base_path_name, "previsto_scatter", k))
    #
    #     val_data_set = validation_set.normalized_data_set_separator(n_steps, goal_row, 12, 3, False, norm_rule="less_one_one")[k]
    #     predicted = mlp.out(
    #         val_data_set,
    #         x_label="{} previsto".format(goal_type.upper()), y_label="{} real".format(goal_type.upper()),
    #         plot_type='plot', save_plot=True, filename=str_template.format(base_path_name, "previsto_line", k))
    #
    #     mlp.save(str_template.format(base_path_name, "ann", k))
    #     try:
    #         r_q_est = r_sqrt(gen_train_sets.normalized_data_set_separator(n_steps, goal_row, 12, 3, False, norm_rule="less_one_one")[k],
    #                          sim)
    #
    #         r_q = r_sqrt(validation_set.normalized_data_set_separator(n_steps, goal_row, 12, 3, False, norm_rule="less_one_one")[k],
    #                      predicted)
    #     except:
    #         r_q_est = 0
    #         r_q = 0
    #
    #     with open("{}/{}".format(base_path_name, "params_{}_mes.txt".format(k)), "wb") as f:
    #         frame = inspect.currentframe()
    #         args, _, _, values = inspect.getargvalues(frame)
    #
    #         for i in args:
    #             f.write(bytes("{}: {}\n".format(i, values[i]), encoding='utf8'))
    #         f.write(bytes("minimum error: {}\n".format(min_error[-1]), encoding='utf8'))
    #
    #         f.write(bytes("r squared estimation: {}\n".format(r_q_est), encoding='utf8'))
    #         f.write(bytes("r squared forecast: {}\n".format(r_q), encoding='utf8'))
    #
    #     with open("{}/{}".format(base_path_name, "r_squared_estimated{}_mes.txt".format(k)), "wb") as f:
    #         f.write(bytes("{};\n".format(r_q_est), encoding='utf8'))
    #
    #     with open("{}/{}".format(base_path_name, "r_squared_forecast{}_mes.txt".format(k)), "wb") as f:
    #         f.write(bytes("{};\n".format(r_q), encoding='utf8'))

# siglas:
#     uir = usina iracemapolis
#     usm = usina sao martinho
#     usc = usina sao carlos
#     ubv = usina boa vista

if __name__ == '__main__':
    # month_ann('atr', 3, 1, [60, 20, 1], "train_rprop", epochs=400)

    shape = [40, 10,1]


    for d in range(9, 0, -1):
        delay = d
        for n_steps in range(0, d):
            # n_steps = n_steps * 5
            for i in range(1, 7):
                n_s = n_steps + i
                try:
                    # month_ann('atr', n_steps, shape, "train_rprop", epochs=400)

                    month_ann('tch', n_s, delay, shape, "train_rprop", epochs=600, show=50, sum_steps=6)
                    # month_ann('tch', n_steps, shape, "train_rprop", epochs=600, show=50, adapt=True)

                    # month_ann('atr', n_steps, shape, "train_ncg")
                    month_ann('tch', n_s, delay, shape, "train_ncg", sum_steps=4)
                    # month_ann('atr', n_steps, shape, "train_gdx", epochs=760)
                    month_ann('tch', n_s, delay, shape, "train_gdx", epochs=900, show=50, sum_steps=6)
                    # month_ann('tch', n_steps, shape, "train_gdx", adapt=True, epochs=900, show=50)
                    # delay = delay - 1
                except Exception as e:
                    print(e)

            delay -= 1


    # for n_steps in range(20, 54):
    #     try:
    #         # month_ann('atr', n_steps, shape, "train_rprop", epochs=400)
    #
    #         month_ann('tch', n_steps, 1, shape, "train_rprop", epochs=600, show=50)
    #         # month_ann('tch', n_steps, shape, "train_rprop", epochs=600, show=50, adapt=True)
    #
    #         # month_ann('atr', n_steps, shape, "train_ncg")
    #         month_ann('tch', n_steps, 1, shape, "train_ncg")
    #         # month_ann('atr', n_steps, shape, "train_gdx", epochs=760)
    #         month_ann('tch', n_steps, 1, shape, "train_gdx", epochs=900, show=50)
    #         # month_ann('tch', n_steps, shape, "train_gdx", adapt=True, epochs=900, show=50)
    #     except Exception as e:
    #         print(e)

