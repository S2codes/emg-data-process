import numpy as np
from scipy.signal import hilbert
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Assuming your data is stored in a CSV file
# Replace 'your_data.csv' with the actual path to your data file
data_path = 'data.csv'

# Load the data
data = np.genfromtxt(data_path, delimiter=',', skip_header=1)

# Separate features (EMG signals) and labels (output)
X = data[:, :-1]
y = data[:, -1]

# Apply envelope detection using Hilbert transform
envelope_data = np.abs(hilbert(X, axis=0))

# Concatenate the envelope data with the original data
X_with_envelope = np.concatenate((X, envelope_data), axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_with_envelope, y, test_size=0.2, random_state=42)

# Train a machine learning model (Random Forest classifier in this example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')
