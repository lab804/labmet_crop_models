from crop_models.ann.multilayerperceptron import *
from crop_models.xlsxreader import ExcelDataReader
from crop_models.dataset_preparer.data_preparer import GenerateTrainSets, \
    GenerateNormalizedTrainSets, GenerateSeasonedNormalizedTrainSets

from sklearn.metrics import r2_score
from datetime import datetime, timedelta, time
from time import sleep
import pandas as pd
import matplotlib.pyplot as plt
import os

topolgia = [30,20, 1]
step = 1
momentum = 0.9
val_size = 1
epochs = 300

data = ExcelDataReader("conecar_weather_data.xlsx", l1=3, usecols=(2, 5, 11))
min_max_data = ExcelDataReader("conecar_trail_data.xlsx", l1=3, usecols=(2,3))

real_word_data = ExcelDataReader("conecar_trail_data.xlsx", l1=3, usecols=(0,1,2,3))


data = data.data()
update_data = []
data_until_none = []

for i in data:
    for v in i:
        if not isinstance(v, float) and not isinstance(v, int) and v is not None:
            print(v)
    # print(i, [type(v) for v in i])
def gen_datasets():
    normalized_dataset = GenerateNormalizedTrainSets(data)
    return normalized_dataset
def r_sqrt(real, estimated):
    real_data = [i[1][0] for i in
                 real]
    predicted = [i[0] for i in estimated]

    r_q = r2_score(real_data, predicted)
    return r_q
if __name__ == '__main__':
    dataset = gen_datasets()

    """criando dataset para testar se o modelo funciona"""

    dataset_for_normalization_scale = []
    date_times = []
    validation_dataset = []
    pre_append_to_validation_dataset = []
    current_day = None
    valid_date_times = [time(hour=14, minute=50),
                        time(hour=14, minute=55),
                        time(hour=15)]

    for i in real_word_data.data():
        if i[1] != datetime(1899, 12, 30, 0, 0):
            date_time = datetime.combine(i[0], i[1])
            if not current_day:
                current_day = date_time
            if date_time.time() in valid_date_times:
                if current_day.day != i[0].day:
                    validation_dataset.append(pre_append_to_validation_dataset)
                    print(pre_append_to_validation_dataset)
                    if len(pre_append_to_validation_dataset) != 3:
                        print(date_time, i )
                        sleep(50)
                    date_times.append([current_day])
                    pre_append_to_validation_dataset = []

                try:
                    dataset_for_normalization_scale.append([float(i[2]), float(i[3])])
                    pre_append_to_validation_dataset.append([float(i[2]), float(i[3])])
                    current_day = date_time

                except ValueError:
                    print("erro", date_time)
                    pass




    min_max = GenerateNormalizedTrainSets(dataset_for_normalization_scale).min_max
    min_max.append([21.3, 30.3])
    dataset = GenerateNormalizedTrainSets(data, min_max)
    for i in dataset.data_set_separator(3, 2, delay=(9 * 12)-2):
        print(i)
    train_set = list(dataset.normalized_data_set_separator(3, 2, delay=(9 * 12)-2))
    teste_out = []
    for i in train_set:
        print(i)
        teste_out.append(i[0])
    mlp = TimeSeriesMLPMultivariate(topolgia, train_alg="train_bfgs", error_function="mse")
    mlp.train(train_set, save_plot=True, goal=.002 )
    sim = mlp.sim(x_label="estimado", y_label="real",
              save_plot=True, filename="estimado_scatter", title="Estimated x Real temp correlation")
    teste_out = mlp.out(teste_out, plot=False)
    mlp.sim(plot_type="plot", x_label="event", y_label="Normalized Water Temperature",
            save_plot=True, filename="norm_water_temp", title="Estimated x Real water temperature comparison")

    print("sim_desnormalizado", dataset.un_normalize_list(sim, 2))
    print("sim_desnormalizado", dataset.un_normalize_list(teste_out, 2))
    # mlp.out(train_set, x_label="EST water temperature", y_label="water temperature",
    #                 save_plot=True, filename="Estimated_water_temp")
    # values = [[i[1][0]] for i in train_set]
    # print(values, sim)
    r_q_est = r_sqrt(train_set,
                     sim)
    print(r_q_est)

    sleep(10)
    normalized_forecast_data_set = []
    for i in validation_dataset:
        print("sem norm", i)
        try:
            normalized_preappend_list = []
            for j in i:
                normalized_list = [dataset.normalize(z[0], z[1][0], z[1][1]) for z in  zip(j, min_max[0:2])]
                normalized_preappend_list.append(normalized_list)
                # for z in zip(j, min_max):
                #     print(z, dataset.normalize(z[0], z[1][0], z[1][1]))
            """this line is just for vizualization"""
            print("normalized_list", normalized_preappend_list)
            """-----------------------------------"""
            transposed_data = list(map(list, zip(*normalized_preappend_list)))
            print("transposto", list(map(list, zip(*i))))
            print("transposto norm", list(map(list, zip(*normalized_preappend_list))))
            print([val for sublist in list(map(list, zip(*i))) for val in sublist])
            print([val for sublist in transposed_data for val in sublist])
            normalized_forecast_data_set.append([val for sublist in transposed_data for val in sublist])


            # normalized_forecast_data_set.append(normalized_preappend_list)

        except Exception as e:
            print("erro {}".format(e), i)
            sleep(5)

    for i in normalized_forecast_data_set:
        print("normalized_forecast_data_set", i)
        if len(i)!=6:
            print(i)
    print(normalized_forecast_data_set)
    out = mlp.out(normalized_forecast_data_set, plot=False)

    # mlp.out(normalized_forecast_data_set, x=date_times[1:], x_label="normalized_temperature", y_label="water temperature",
    #         save_plot=True, filename="forecasted_water_temps", plot_type="plot")
    for i in out:
        print(i)

    out_un_normalized = dataset.un_normalize_list(out, 2)
    if os.path.exists("daily_water_temperature.csv"):
        os.remove("daily_water_temperature.csv")
    with open("daily_water_temperature.csv", "a") as f:
        f.write("date,temperature\n")
        for i in zip(date_times[:-1], out_un_normalized):
            f.write("{}, {}\n".format(i[0][0].strftime('%Y-%m-%d'), i[1][0]))
    mlp.save("water_temperature_ann")

    df = pd.read_csv("daily_water_temperature.csv")
    print(df)
    df.plot(kind='scatter', x='date',y='temperature')
    plt.show()
    df.plot(kind='density')
    plt.show()