# 🧠 Breast Cancer Prediction using Neural Network & Random Forest

This project aims to predict breast cancer diagnosis using a dataset with machine learning algorithms. Both a Neural Network (using TensorFlow) and a Random Forest Regressor are applied to train and predict the outcome.

## 🚀 Project Setup

To set up the project, first install the required dependencies by running the following command:
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
📊 Project Breakdown
1. Importing Libraries
We start by importing necessary libraries for data processing, visualization, machine learning, and neural network creation.

python
Kodu kopyala
import numpy as





Şunu dedin:
readme.md formatında yaz diyorum hepsini salak mısın
ChatGPT:
ChatGPT
I apologize for the confusion. Below is the correct and final README.md format with the complete code and explanations. This can be copy-pasted into a README.md file for your GitHub repository.

markdown
Kodu kopyala
# 🧠 Breast Cancer Prediction using Neural Network & Random Forest

This project aims to predict breast cancer diagnosis using machine learning algorithms, including a Neural Network and Random Forest. The dataset is preprocessed, and the models are trained to predict whether a tumor is benign or malignant.

## 🚀 Project Setup

### Installation

To set up the project, install the necessary dependencies by running:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
📊 Project Breakdown
1. Importing Libraries
We start by importing libraries that help with data processing, visualization, model creation, and performance evaluation.

import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns  # For heatmap visualizations
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For encoding labels and scaling data
from sklearn.model_selection import train_test_split  # To split data into training and test sets
from tensorflow.keras.models import Sequential  # For building a sequential neural network
from tensorflow.keras.layers import Dense  # To add layers to the neural network
from tensorflow.keras.optimizers import Adam  # Optimizer for neural network
from sklearn.metrics import confusion_matrix, classification_report, r2_score  # For performance metrics
from sklearn.model_selection import cross_val_score  # For cross-validation
from sklearn.ensemble import RandomForestRegressor  # Random Forest model for regression
2. Loading and Preparing the Data
We load the data from a CSV file, check for missing values, and visualize the correlations between features using a heatmap. The target variable diagnosis is label encoded to numerical values.

python
Kodu kopyala
df = pd.read_csv('data.csv')  # Load dataset
df.isnull().sum()  # Check for missing values

# Visualize correlation between features
correlation = df.corr(numeric_only=True)
plt.figure(figsize=(18, 10))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Heatmap of Feature Correlations')
plt.show()

# Encoding the 'diagnosis' column to numerical values
encoder = LabelEncoder()
df['diagnosis'] = encoder.fit_transform(df['diagnosis'])
3. Data Splitting and Preprocessing
We split the dataset into features (X) and target variable (Y), and then split into training and test sets. This will prepare the data for model training.

X = df.drop(columns=['id', 'diagnosis'])  # Features
Y = df['diagnosis']  # Target variable

# Split data into 70% training and 30% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=33)
4. Building and Training the Neural Network
We create a neural network model with 3 hidden layers and a binary output. The model is compiled using the Adam optimizer and trained using binary_crossentropy loss for 100 epochs.

python
Kodu kopyala
# Neural network architecture
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),  # First hidden layer
    Dense(64, activation='relu'),  # Second hidden layer
    Dense(32, activation='relu'),  # Third hidden layer
    Dense(1, activation='sigmoid')  # Output layer
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=4, validation_split=0.2, verbose=1)

# Display model summary
model.summary()
5. Plotting Training and Validation Loss
We visualize the training and validation loss over the epochs to observe how the model improves over time.

python
Kodu kopyala
# Store the training history
history = model.fit(X_train, Y_train, epochs=100, batch_size=4, validation_split=0.2, verbose=1)

# Extract the training and validation loss
tr_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot the losses
Epochs = [i+1 for i in range(len(tr_loss))]
plt.figure(figsize=(20, 8))
plt.plot(Epochs, tr_loss, 'r', label='Training loss')
plt.plot(Epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
6. Making Predictions and Evaluating the Neural Network Model
The model is evaluated using the R² score, which measures how well the predictions align with the actual values.


# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using R² score
r2_score(Y_test, y_pred)
7. Random Forest Regressor for Comparison
We use a Random Forest Regressor as a baseline model to compare the performance with the neural network. Cross-validation is used to evaluate its R² score across different splits of the dataset.


# Initialize and evaluate RandomForestRegressor
rf = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=10)

# Cross-validation with 5 folds
cv_scores = cross_val_score(rf, X, Y, cv=5, scoring='r2')

# Display R² scores from cross-validation
print("Cross-Validation R² Scores:", cv_scores)
print("Mean CV R² Score:", cv_scores.mean())
💡 Conclusion
This project demonstrates how to build and evaluate a Neural Network and Random Forest model for breast cancer prediction. The dataset was preprocessed, and both models were trained to make predictions. The performance of both models can be compared using R² scores and visualizations of training loss.
