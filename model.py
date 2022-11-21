import pandas as pd
from math import pi, sin
from keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


df = pd.read_csv('data_trc.csv')

sine_m = []
sine_y = []

for i in range(0, 30):
    sine_m.append(0.5 * sin(i * pi / 14.5) + 0.5)

for i in range(0, 365):
    sine_y.append(0.5 * sin(i * pi / 182.5) + 0.5)

sales = {}
sine350x = sine_m * 350 
sine4x = sine_y * 4 

for i in range(len(df)):
    if df['date'][i] in list(sales.keys()):
        sales[df['date'][i]] += df['sales'][i]
    else:
        sales[df['date'][i]] = df['sales'][i]

sales = pd.DataFrame({'dates': list(sales.keys()), 'sine_y': sine4x[:len(sales.keys())], 
                        'sine_m': sine350x[:len(sales.keys())], 'sales': [sales[k] for k in list(sales.keys())]})

MAX = max(sales['sales'])
MIN = min(sales['sales'])
sales_norm = sales.copy()
sales_norm['sales'] = (sales_norm['sales'] - MIN) / (MAX - MIN)

length = 1
time_series = TimeseriesGenerator([[sales_norm['sales'][i - 1], sales['sine_y'][i], sales['sine_m'][i]]  for i in range(1, len(sales))], 
                                    sales_norm['sales'][1:], length=length, batch_size=len(sales_norm))
input, target = time_series[0]

print(input.shape)
print(target.shape)

train_portion = 0.7
tp = int(train_portion * len(input))

input_t = input[:tp]
input_v = input[tp:]

target_t = target[:tp]
target_v = target[tp:]

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(1, 3)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))

model.compile(loss=tf.keras.losses.MeanSquaredError(), 
                optimizer=tf.keras.optimizers.Adam(), 
                metrics=[tf.keras.metrics.MeanAbsoluteError()])
earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')
history = model.fit(input_t, target_t, batch_size=3, epochs=200, 
                    shuffle=False, validation_data=(input_v, target_v), 
                    callbacks=[earlyStop])

model.save('model')