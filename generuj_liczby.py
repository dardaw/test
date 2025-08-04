import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Pobieranie danych historycznych
# Zakładamy, że dane są w pliku CSV z kolumnami: data, num1, num2, num3, num4, num5, num6
data = pd.read_csv('file.csv')

# 2. Przetwarzanie danych
# Wybieramy tylko kolumny z numerami
numbers = data[['num1', 'num2', 'num3', 'num4', 'num5']].values

# Normalizacja danych do zakresu [0, 1] (dla puli 1-49)
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

# Podział na zbiory treningowy i testowy (80% trening, 20% test)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 3. Budowanie i trenowanie modelu LSTM
model = Sequential()
model.add(LSTM(240, return_sequences=True, input_shape=(seq_length, 5)))
model.add(Dropout(0.2))
model.add(LSTM(240, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(240))
model.add(Dropout(0.2))
model.add(Dense(5, activation='sigmoid'))  # Wyjście: 6 liczb w zakresie [0,1]

model.compile(optimizer='adam', loss='mse')

# Trenowanie modelu z early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=300, batch_size=100, validation_data=(X_test, y_test), callbacks=[early_stopping])

# 4. Przewidywanie przyszłych wyników
# Użyj ostatnich 10 losowań do przewidzenia następnego
last_sequence = numbers_scaled[-seq_length:]
last_sequence = np.expand_dims(last_sequence, axis=0)

prediction_scaled = model.predict(last_sequence)
prediction = scaler.inverse_transform(prediction_scaled)  # Przekształcenie z powrotem do skali 1-49

# Zaokrąglenie do najbliższych liczb całkowitych i zapewnienie unikalności
prediction = np.round(prediction).astype(int)
prediction = np.clip(prediction, 1, 42)  # Ograniczenie do zakresu 1-49

# Usuwanie duplikatów i zapewnienie dokładnie 6 liczb
unique_numbers = []
seen = set()
for num in prediction.flatten():
    if num not in seen and len(unique_numbers) < 5:
        unique_numbers.append(num)
        seen.add(num)

# Jeśli jest mniej niż 6 liczb, wybieramy najbliższe unikalne wartości
while len(unique_numbers) < 5:
    for i in range(1, 43):
        if i not in seen and len(unique_numbers) < 5:
            unique_numbers.append(i)
            seen.add(i)

# Sortowanie liczb dla czytelności
unique_numbers = sorted(unique_numbers)

print("Przewidywane liczby dla następnego losowania:", unique_numbers) 