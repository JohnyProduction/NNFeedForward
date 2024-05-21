# Install to use the combination of pytorch and scikit-learn
!pip install skorch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load dataset
df = pd.read_csv('./House_Rent_Dataset.csv')

# Separate target and features
target = pd.DataFrame(df['Rent'])
df.drop('Rent', axis=1, inplace=True)

# Encode categorical features
for col in df[['Floor', 'Area Type', 'Area Locality', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']]:
    df[col] = pd.Categorical(df[col]).codes

# Convert date to numeric
df['Posted On'] = pd.to_datetime(df['Posted On'])
df['Posted On'] = df['Posted On'].astype('int64') // 10**10

# Scale numeric features
scaler = StandardScaler()
numeric_cols = ['Posted On', 'BHK', 'Size', 'Floor', 'Bathroom']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Check the first few rows
print(df.head())
print(target.head())

# Plot histograms for features and target
df_hist = df.copy()
df_hist['Rent'] = target

# Plot histograms for all columns
df_hist.hist(bins=20, figsize=(20, 15))
plt.tight_layout()
plt.show()

# Create a scatterplot matrix using Seaborn
sns.pairplot(df_hist, diag_kind='kde')
plt.show()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df.values, target.values, test_size=0.2, random_state=42)

# Convert Rent values to a binary class for classification (for example, high rent vs. low rent)
median_rent = target.median()[0]
y_train_class = (y_train >= median_rent).astype(int).ravel()  # Ensure 1D
y_test_class = (y_test >= median_rent).astype(int).ravel()  # Ensure 1D

# Define the neural network architecture with flexible layers
class HouseRentNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(HouseRentNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Define the input and output dimensions
input_dim = X_train.shape[1]
output_dim = 2

# Create the neural network
net = NeuralNetClassifier(
    module=HouseRentNN,
    max_epochs=20,
    lr=0.1,
    optimizer=optim.Adam,
    criterion=nn.CrossEntropyLoss,
    iterator_train__shuffle=True,
    verbose=0,
    module__input_dim=input_dim,
    module__output_dim=output_dim
)

# Define the parameter grid for GridSearchCV
params = {
    'lr': [0.01,0.2],
    'max_epochs': [10,30],
    'module__hidden_dims': [
        [32],[64]
    ],
    'optimizer__weight_decay': [0, 1e-4, 1e-2]
}

# Perform grid search
gs = GridSearchCV(net, params, refit=True, cv=3, scoring='accuracy')
gs.fit(X_train.astype(np.float32), y_train_class)

# Print best parameters and best score
print("Best parameters found: ", gs.best_params_)
print("Best cross-validation accuracy: ", gs.best_score_)

# Print all results
results = pd.DataFrame(gs.cv_results_)
print(results)

# Evaluate the best model on the test set
best_model = gs.best_estimator_
test_accuracy = best_model.score(X_test.astype(np.float32), y_test_class)
print("Test set accuracy: ", test_accuracy)

# Plot results
plt.figure(figsize=(12, 6))

# Plot Validation Accuracy vs Learning Rate
plt.subplot(1, 3, 1)
for max_epoch in params['max_epochs']:
    subset = results[results['param_max_epochs'] == max_epoch]
    plt.plot(subset['param_lr'], subset['mean_test_score'], label=f'epochs={max_epoch}')
plt.xlabel('Learning Rate')
plt.ylabel('Mean Validation Accuracy')
plt.title('Validation Accuracy vs Learning Rate')
plt.legend()

# Plot Validation Accuracy vs Number of Epochs
plt.subplot(1, 3, 2)
for lr in params['lr']:
    subset = results[results['param_lr'] == lr]
    plt.plot(subset['param_max_epochs'], subset['mean_test_score'], label=f'lr={lr}')
plt.xlabel('Number of Epochs')
plt.ylabel('Mean Validation Accuracy')
plt.title('Validation Accuracy vs Number of Epochs')
plt.legend()

# Plot Validation Accuracy vs Hidden Layer Configuration
plt.subplot(1, 3, 3)
hidden_dims_str = [str(hd) for hd in params['module__hidden_dims']]
hidden_dims_mapping = {str(hd): i for i, hd in enumerate(params['module__hidden_dims'])}
results['hidden_dims_str'] = results['param_module__hidden_dims'].astype(str)
for weight_decay in params['optimizer__weight_decay']:
    subset = results[results['param_optimizer__weight_decay'] == weight_decay]
    plt.plot([hidden_dims_mapping[hd] for hd in subset['hidden_dims_str']], subset['mean_test_score'], label=f'wd={weight_decay}')
plt.xlabel('Hidden Layer Configuration')
plt.xticks(range(len(hidden_dims_str)), hidden_dims_str, rotation=90)
plt.ylabel('Mean Validation Accuracy')
plt.title('Validation Accuracy vs Hidden Layer Configuration')
plt.legend()

plt.tight_layout()
plt.show()

# Feature importance using permutation importance
perm_importance = permutation_importance(best_model, X_test.astype(np.float32), y_test_class, n_repeats=10, random_state=42)
feature_importance = perm_importance.importances_mean
feature_names = df.columns

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_importance)), feature_importance, align='center')
plt.yticks(range(len(feature_importance)), feature_names)
plt.xlabel('Permutation Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.show()

# Confusion Matrix
y_pred = best_model.predict(X_test.astype(np.float32))
cm = confusion_matrix(y_test_class, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
y_proba = best_model.predict_proba(X_test.astype(np.float32))
fpr, tpr, _ = roc_curve(y_test_class, y_proba[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Classification Report
print(classification_report(y_test_class, y_pred))

# Plot training loss over epochs for the best model
plt.figure(figsize=(8, 5))
plt.plot(best_model.history[:, 'train_loss'], label='Train Loss')
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.grid(True)
plt.show()
