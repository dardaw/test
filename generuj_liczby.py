import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout
batch_size = 100
ile_generowac = 300 #powinno być 300

df = pd.read_csv(r"/home/dardaw/test/file.csv")
#df = pd.read_csv(r"C:\xampp\htdocs\test\file.csv")

df.drop(['Game', 'Date'], axis=1, inplace=True)

scaler = StandardScaler().fit(df.values)
transformed_dataset = scaler.transform(df.values)
transformed_df = pd.DataFrame(data=transformed_dataset, index=df.index)

number_of_rows = df.values.shape[0]
window_length = 7
number_of_features = df.values.shape[1]
X = np.empty([ number_of_rows - window_length, window_length, number_of_features], dtype=float)
y = np.empty([ number_of_rows - window_length, number_of_features], dtype=float)

for i in range(0, number_of_rows-window_length):
    X[i] = transformed_df.iloc[i : i+window_length, 0 : number_of_features]
    y[i] = transformed_df.iloc[i+window_length : i+window_length+1, 0 : number_of_features]

model = Sequential()
# Adding the input layer and the LSTM layer
model.add(Bidirectional(LSTM(240,
                        input_shape = (window_length, number_of_features),
                        return_sequences = True)))
# Adding a first Dropout layer
model.add(Dropout(0.2))
# Adding a second LSTM layer
model.add(Bidirectional(LSTM(240,
                        input_shape = (window_length, number_of_features),
                        return_sequences = True)))
# Adding a second Dropout layer
model.add(Dropout(0.2))
# Adding a third LSTM layer
model.add(Bidirectional(LSTM(240,
                        input_shape = (window_length, number_of_features),
                        return_sequences = True)))
# Adding a fourth LSTM layer
model.add(Bidirectional(LSTM(240,
                        input_shape = (window_length, number_of_features),
                        return_sequences = False)))
# Adding a fourth Dropout layer
model.add(Dropout(0.2))
# Adding the first output layer
model.add(Dense(59))
# Adding the last output layer
model.add(Dense(number_of_features))

from tensorflow import keras
from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.0001), loss ='mse', metrics=['accuracy'])
model.fit(x=X, y=y, batch_size=100, epochs=ile_generowac, verbose=2)

to_predict = df.tail(8) #tu bylo 8

#to_predict.drop([to_predict.index[-1]],axis=0, inplace=True) #tu bylo -1
to_predict = np.array(to_predict)
print(to_predict)
scaled_to_predict = scaler.transform(to_predict)

y_pred = model.predict(np.array([scaled_to_predict]))
print('Cos ',to_predict)
print("Przewidywane liczby to:", scaler.inverse_transform(y_pred).astype(int)[0])

prediction = df.tail(1)
prediction = np.array(prediction)
print("Aktualne liczby z pliku to:", prediction[0])
