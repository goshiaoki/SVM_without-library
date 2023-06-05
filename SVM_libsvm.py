import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from libsvm.svmutil import svm_train, svm_predict, svm_parameter, svm_problem

model_type = "soft"  # "soft" or "hard"

# Load the data
data = pd.read_csv("./data.csv", index_col=0)

# Prepare the input data
X = data.iloc[:,2:].values
y = data.iloc[:,1].values
print(X.shape)

# Replace "B" with 1 and "M" with -1
y[y == "B"] = 1
y[y == "M"] = -1
y = y.astype(int)

# Standardize the input data
standard_scaler = StandardScaler()
standardized_X = standard_scaler.fit_transform(X)

# Split the data into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(standardized_X, y, stratify=y, random_state=0)


# Convert y to double
train_y = train_y.astype('double')
test_y = test_y.astype('double')

# Create problem and parameter objects
prob = svm_problem(train_y, train_X)

if model_type == "soft":
  param = svm_parameter('-t 0 -c 1 -b 1')  # -t 0 for linear kernel, -c 1 for cost, -b 1 for probability estimates
  print("Soft Margin SVM with libsvm")
elif model_type == "hard":
  param = svm_parameter('-t 0 -c 1000000 -b 1')  # -t 0 for linear kernel, -c 1000000 for cost, -b 1 for probability estimates
  print("Hard Margin SVM with libsvm")

# Train the model
model = svm_train(prob, param)

# Predict the labels of the test data
p_label, p_acc, p_val = svm_predict(test_y, test_X, model)

# The accuracy is stored in p_acc[0]
print("Accuracy:", p_acc[0])

# The weights of the model are stored in model.coef[0]
# Get the feature names
feature_names = data.columns[2:]
# Print the weights with the feature names
for feature, weight in zip(feature_names, model.get_sv_coef()):
    print(f"{feature}: {weight}")

# The intercept is stored in model.rho[0]
print("Intercept (b):", -model.rho[0])