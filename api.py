from tensorflow.keras.models import load_model
from flask import Flask, jsonify, request
import pandas as pd
from math import pi, sin


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

app = Flask(__name__)
app.config['PORT'] = 1234
app.config['HOST'] = '0.0.0.0'

model = load_model('model')

@app.route('/api/predict', methods=['GET'])
def predict():
    date = request.args.get('date')

    d = date[:2]
    m = date[2:4]
    y = date[4:]
    october1st = 274

    if int(d) > 31 or int(d) < 1:
        return jsonify("Wrong day format!")

    if int(m) != 10:
        return jsonify("Predictions are only available for October!")

    if int(y) != 2019:
        return jsonify("Predictions are only available for 2019!")

    last_sale = [[sales_norm['sales'].iloc[-1]]]
    sine_y_pred = sales['sine_y'][october1st + int(d) - 1]
    sine_m_pred = sales['sine_m'][int(d) - 1]

    for _ in range(int(d)):
        last_sale = model.predict([[float(last_sale[0][0]), sine_y_pred, sine_m_pred]])

    result = {'Date': f"{d}/{m}/{y}",
                'Sales': last_sale[0][0] * (MAX - MIN) + MIN}

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host=app.config['HOST'], port=app.config['PORT'])