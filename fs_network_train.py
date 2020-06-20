import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

from constants import *
from interpolator import Interpolator

if __name__ == "__main__":
    TRAIN_TO_EVAL_RATIO = 4  # Lines of training data per 1 line of eval data

    # tf.autograph.set_verbosity(0)
    # tf.device('/gpu:0')
    # tf.compat.v1.disable_eager_execution()

    data = pd.read_csv('logs/' + 'FSG/1592212832.csv')
    #data = data.iloc[::-1]
    data_size = len(data)
    data_generator = data.iterrows()


    def get_data_row():
        data_row = next(data_generator)[1]
        return {"state": eval(data_row["State"]),
                "action": eval(data_row["Action"]),
                "reward": data_row["Reward"],
                "next_state": eval(data_row["NextState"]),
                "done": data_row["Done"]}


    def create_model():
        model = Sequential()
        model.add(Dense(32, activation='sigmoid', input_shape=INPUT_2D_SHAPE))
        model.add(Flatten())
        # for i in range(4):
        #    model.add(Dense(2**(8-i), activation='relu'))
        # model.add(Dense(125, activation='sigmoid'))
        model.add(Dense(75, activation='sigmoid'))  # LOOK AT SIGMOID SOME MORE
        #model.add(Dense(125, activation='sigmoid'))
        model.add(Dense(OUTPUT_1D_SHAPE, activation='linear'))
        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model


    eval_data_list = [get_data_row() for _ in range(int(data_size / (TRAIN_TO_EVAL_RATIO + 1)))]
    train_data_list = [get_data_row() for _ in range(int(data_size / (TRAIN_TO_EVAL_RATIO + 1) * TRAIN_TO_EVAL_RATIO))]
    model = create_model()


    def calculate_qs(data_list):
        qualities_list = model.predict(np.array([data_row["state"] for data_row in data_list]))
        next_qualities_list = model.predict(np.array([data_row["next_state"] for data_row in data_list]))

        for data_row_index in range(len(data_list)):
            data_list[data_row_index]["qualities"] = qualities_list[data_row_index]
            data_list[data_row_index]["next_qualities"] = next_qualities_list[data_row_index]


    def get_x_y(data_list):
        interpolator = Interpolator()
        interpolator.set_u(ACTIONS)
        x = []
        y = []
        for data_row in data_list:
            new_q = data_row["reward"]
            if not data_row["done"]:
                new_q += DISCOUNT * np.max(data_row["next_qualities"])
            interpolator.set_q(data_row["qualities"])
            interpolator.update_function(data_row["action"], new_q)
            x.append(data_row["state"])
            y.append(interpolator.get_q())
        return x, y


    calculate_qs(eval_data_list)
    calculate_qs(train_data_list)
    train_x, train_y = get_x_y(train_data_list)
    eval_x, eval_y = get_x_y(eval_data_list)


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


    # start = time.time()

    # for _ in range(10):
    model.fit(np.array(train_x), np.array(train_y),
              batch_size=64,
              epochs=3,
              callbacks=[MyCustomCallback()])
    model.save('models/__' + TRACK_FILE.split('.')[0] + '_trained_' + str(round(time.time())) + '.model')
    # end = time.time()
    # print(end - start)
    # print(len(eval_data_list))
    # agent = DQNAgent()
