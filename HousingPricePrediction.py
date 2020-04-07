import pandas as pd
housing= pd.read_csv('cal_housing_clean.csv')

x_data=housing.drop(['medianHouseValue'],axis=1)
y_data=housing['medianHouseValue']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=101)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(x_train)

x_train=pd.DataFrame(data=scaler.transform(x_train), columns=x_train.columns, index=x_train.index)

x_test=pd.DataFrame(data=scaler.transform(x_test), columns=x_test.columns, index=x_test.index)

import tensorflow as tf

hma= tf.feature_column.numeric_column('housingMedianAge')
tr = tf.feature_column.numeric_column('totalRooms')
tb = tf.feature_column.numeric_column('totalBedrooms')
po = tf.feature_column.numeric_column('population')
ho = tf.feature_column.numeric_column('households')
mi = tf.feature_column.numeric_column('medianIncome')


feat_cols=[hma, tr, tb, po, ho, mi]


input_func=tf.estimator.inputs.pandas_input_fn(x=x_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)

model=tf.estimator.DNNRegressor(hidden_units=[6,6,6,3], feature_columns=feat_cols)

model.train(input_fn=input_func, steps=10000)

pred_input_func=tf.estimator.inputs.pandas_input_fn(x=x_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)

pred_gen=model.predict(pred_input_func)

predictions = list(pred_gen)

final_preds = []
for pred in predictions:
    final_preds.append(pred['predictions'])
    
from sklearn.metrics import mean_squared_error

print(mean_squared_error(y_test,final_preds)**0.5)