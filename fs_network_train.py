import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

from constants import *
from interpolator import Interpolator
from os import path, makedirs, chmod

if __name__ == "__main__":
    file_names = ['logs/FSG/1592749239/data.csv',
                  'logs/FSG/1592687970/data.csv',
                  'logs/FSG/1592677468/data.csv',
                  'logs/FSG/1592665262/data.csv',
                  'logs/FSG/1592664316/data.csv',
                  'logs/FSG/1592656783/data.csv',
                  'logs/FSG/1592648977/data.csv',
                  'logs/FSG/1592644339/data.csv',
                  'logs/FSG/1592648977/data.csv'
                  ]
    data_files = []

    for file_name in file_names:
        data_files.append(pd.read_csv(file_name, index_col=None, header=0))

    data = pd.concat(data_files, axis=0, ignore_index=True)

    # data = pd.read_csv('logs/' + 'FSG/1592212832.csv')
    # data = data.iloc[::-1]
    data_size = len(data)
    data_generator = data.iterrows()


    def get_data_row():
        data_row = next(data_generator)[1]
        return {"state": eval(data_row["State"]),
                "action": eval(data_row["Action"]),
                "reward": data_row["Reward"],
                "next_state": eval(data_row["NextState"]),
                "done": data_row["Done"]}


    def create_state_model():
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=(3, 2)))
        model.add(Flatten())
        model.add(Dense(75, activation='relu'))
        model.add(Dense(4, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


    def create_quality_model():
        model = Sequential()
        model.add(Dense(32, activation='relu', input_shape=(3, 2)))
        model.add(Flatten())
        for i in range(4):
            model.add(Dense(2 ** (8 - i), activation='relu'))
        # model.add(Dense(125, activation='sigmoid'))
        # model.add(Dense(75, activation='relu'))
        model.add(Dense(1, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


    # eval_data_list = [get_data_row() for _ in range(int(data_size / (TRAIN_TO_EVAL_RATIO + 1)))]
    train_data_list = [get_data_row() for _ in range(data_size)]
    state_model = create_state_model()
    quality_model = create_quality_model()

    '''def calculate_qs(data_list):
        qualities_list = model.predict(np.array([data_row["state"] for data_row in data_list]))
        next_qualities_list = model.predict(np.array([data_row["next_state"] for data_row in data_list]))

        for data_row_index in range(len(data_list)):
            data_list[data_row_index]["qualities"] = qualities_list[data_row_index]
            data_list[data_row_index]["next_qualities"] = next_qualities_list[data_row_index]
    '''


    def get_x_y(data_list):
        # interpolator = Interpolator()
        # interpolator.set_u(ACTIONS)
        state_x = []
        state_y = []
        quality_x = []
        quality_y = []
        for data_row in data_list:
            # new_q = data_row["reward"]
            # if not data_row["done"]:
            #    new_q += DISCOUNT * np.max(data_row["next_qualities"])
            # interpolator.set_q(data_row["qualities"])
            # interpolator.update_function(data_row["action"], new_q)
            state_x.append(list(data_row["state"]) + [list(data_row["action"])])
            state_y.append(np.array(data_row["next_state"]).flatten())  # interpolator.get_q())
            quality_x.append(list(data_row["state"]) + [list(data_row["action"])])
            quality_y.append(data_row["reward"] if data_row["reward"] > 0 else -1)
        return state_x, state_y, quality_x, quality_y


    # calculate_qs(eval_data_list)
    # calculate_qs(train_data_list)
    state_train_x, state_train_y, quality_train_x, quality_train_y = get_x_y(train_data_list)
    # eval_x, eval_y = get_x_y(eval_data_list)

    '''
    class MyCustomCallback(tf.keras.callbacks.Callback):
        def on_test_begin(self, logs=None):
            global eval_x
            global eval_y
            calculate_qs(eval_data_list)
            eval_x, eval_y = get_x_y(eval_data_list)

        def on_train_begin(self, logs=None):
            global train_x
            global train_y
            calculate_qs(train_data_list)
            train_x, train_y = get_x_y(train_data_list)
    '''

    # start = time.time()

    # for _ in range(10):
    current_time = str(round(time.time()))
    if not path.isdir(f'models/{current_time}'):
        makedirs(f'models/{current_time}')
        chmod(f'models/{current_time}', 0o664)

    state_model.fit(np.array(state_train_x), np.array(state_train_y),
                    batch_size=64,
                    epochs=3,
                    validation_split=0.2)
    state_model.save(f'models/{current_time}/__trained_state.model')
    chmod(f'models/{current_time}/__trained_state.model', 0o664)
    quality_model.fit(np.array(quality_train_x), np.array(quality_train_y),
                      batch_size=64,
                      epochs=5,
                      validation_split=0.2)
    quality_model.save(f'models/{current_time}/__trained_quality.model')
    chmod(f'models/{current_time}/__trained_quality.model', 0o664)
    # end = time.time()
    # print(end - start)
    # print(len(eval_data_list))
    # agent = DQNAgent()
