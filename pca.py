import numpy as np
import matplotlib.pyplot as plt

# Original matrix X
X = np.array([
    [2, 5],
    [4, 2],
    [6, 3],
    [3, 4],
    [5, 1]
])

# 1. Compute the standardized matrix Z
means = np.mean(X, axis=0)
stds = np.std(X, axis=0, ddof=1)  # using ddof=1 for sample standard deviation
Z = (X - means) / stds

# 2. Compute the covariance matrix S (sample covariance)
S = np.cov(X, rowvar=False)

# 3. Compute the correlation matrix R from S
std_devs = np.sqrt(np.diag(S))
outer_std = np.outer(std_devs, std_devs)
R = S / outer_std

# 4. Compute the eigen-decomposition of S (or equivalently, R)
eigenvalues, eigenvectors = np.linalg.eig(S)
# Form a diagonal matrix D from the eigenvalues
D = np.diag(eigenvalues)
U = eigenvectors  # Columns of U are the eigenvectors (each of unit length)

# 5. Compute the matrix C = Z * U
C = Z @ U

# Display the computed matrices
print("Standardized matrix Z:")
print(Z)
print("\nCovariance matrix S:")
print(S)
print("\nCorrelation matrix R:")
print(R)
print("\nEigenvalue matrix D (diagonal of eigenvalues):")
print(D)
print("\nEigenvector matrix U (each column is an eigenvector of unit length):")
print(U)
print("\nMatrix C = Z * U:")
print(C)

# Plotting the original matrix X and the transformed matrix C
plt.figure(figsize=(12, 5))

# Plot the original data X
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], color='blue', marker='o')
plt.title("Original Matrix X")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)

# Plot the transformed data C
plt.subplot(1, 2, 2)
plt.scatter(C[:, 0], C[:, 1], color='red', marker='^')
plt.title("Transformed Matrix C = Z * U")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)

plt.tight_layout()
plt.show()
