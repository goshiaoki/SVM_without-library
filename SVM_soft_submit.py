import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from cvxopt import matrix, solvers

# Load the data
data = pd.read_csv("./data.csv", index_col=0)

# Prepare the input data
X = data.iloc[:,2:].values
y = data.iloc[:,1].values

# Replace "B" with 1 and "M" with -1
y[y == "B"] = 1
y[y == "M"] = -1
y = y.astype(int)

# Standardize the input data
standard_scaler = StandardScaler()
standardized_X = standard_scaler.fit_transform(X)

# Split the data into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(standardized_X, y, stratify=y, random_state=0)

# Our SVM class
class SVM:
    def __init__(self, C=1.0):
        self.C = C
        self.weights = None
        self.intercept = None

    def fit(self, X, y):
        num_samples, num_X = X.shape

        # P = X^T X
        K = np.dot(X, X.T)
        P = matrix(np.outer(y, y) * K)
        # q = -1 (1xN)
        q = matrix(-np.ones((num_samples, 1)))
        # A = y^T 
        A = matrix(y.astype('double'), (1, num_samples))
        # b = 0 
        b = matrix(0.0)

        # -1 (NxN)
        G1 = np.diag(-np.ones(num_samples))

        # 1 (NxN)
        G2 = np.identity(num_samples)

        G = matrix(np.vstack((G1, G2)))
        h = matrix(np.hstack((np.zeros(num_samples), np.ones(num_samples) * self.C)))
        # solve QP problem
        solution = solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        alpha = np.ravel(solution['x'])

        # Support vectors have non zero lagrange multipliers
        sv = alpha > 1e-5
        ind = np.arange(len(alpha))[sv]
        self.alpha = alpha[sv]
        self.support_vectors = X[sv]
        self.support_vector_y = y[sv]

        # Calculate intercept
        self.intercept = 0
        for n in range(len(self.alpha)):
            self.intercept += self.support_vector_y[n]
            self.intercept -= np.sum(self.alpha * self.support_vector_y * K[ind[n], sv])
        self.intercept /= len(self.alpha)

        # Calculate weights
        self.weights = np.zeros(num_X)
        for n in range(len(self.alpha)):
            self.weights += self.alpha[n] * self.support_vector_y[n] * self.support_vectors[n]

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.intercept).astype(int)


# Create an instance of our SVM and fit the training data
svm = SVM(C=1.0)
svm.fit(train_X, train_y)

# Print out the results
print("Hard Margin SVM with Quadratic Programming")

# Calculate and print out the accuracy
predicted_y = svm.predict(test_X)
accuracy = accuracy_score(test_y, predicted_y)
print("Accuracy:", accuracy)

# Get the feature names
feature_names = data.columns[2:]
# Print the weights with the feature names
for feature, weight in zip(feature_names, svm.weights):
    print(f"{feature}: {weight}")

print("Intercept (b):", svm.intercept)

duality_gap = np.dot(svm.weights, svm.weights) / 2 + svm.C * np.sum(np.maximum(0, 1 - train_y * (np.dot(train_X, svm.weights) + svm.intercept)))
print("Duality gap:", duality_gap)