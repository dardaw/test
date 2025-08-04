import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Pobieranie danych historycznych
# Zakładamy, że dane są w pliku CSV z kolumnami: data, num1, num2, num3, num4, num5, num6
data = pd.read_csv('superenalotto_history.csv')

# 2. Przetwarzanie danych
# Wybieramy tylko kolumny z numerami
numbers = data[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']].values

# Normalizacja danych do zakresu [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
numbers_scaled = scaler.fit_transform(numbers)

# Tworzenie sekwencji: użycie ostatnich 10 losowań do przewidzenia następnego
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X, y = create_sequences(numbers_scaled, seq_length)

# Podział na zbiory treningowy i testowy
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 3. Budowanie i trenowanie modelu LSTM
model = Sequential()
model.add(LSTM(240, return_sequences=True, input_shape=(seq_length, 6)))
model.add(Dropout(0.2))
model.add(LSTM(240, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(240))
model.add(Dropout(0.2))
model.add(Dense(6, activation='sigmoid'))

model.compile(optimizer='adam', loss='mse')

# Trenowanie modelu z early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=300, batch_size=100, validation_data=(X_test, y_test), callbacks=[early_stopping])

# 4. Przewidywanie przyszłych wyników
# Użyj ostatnich 10 losowań do przewidzenia następnego
last_sequence = numbers_scaled[-seq_length:]
last_sequence = np.expand_dims(last_sequence, axis=0)

prediction_scaled = model.predict(last_sequence)
prediction = scaler.inverse_transform(prediction_scaled)

# Zaokrąglenie do najbliższych liczb całkowitych
prediction = np.round(prediction).astype(int)

# Korekta: upewnij się, że liczby są unikalne i w zakresie 1-90
prediction = np.clip(prediction, 1, 90)
prediction = np.unique(prediction)
while len(prediction) < 6:
    new_number = np.random.randint(1, 91)
    if new_number not in prediction:
        prediction = np.append(prediction, new_number)

print("Przewidywane liczby dla następnego losowania:", prediction) 