import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# 1. Original Data
# -----------------------
X = np.array([
    [1396,   90, 182,  940, 407, 169],  # Honda
    [1721,   92, 180,  965, 405, 169],  # Renault 19
    [1580,   83, 170,  930, 396, 170],  # Fiat Tipo
    [1769,   90, 180, 1100, 405, 168],  # Citroen ZX
    [2996,  188, 224, 1700, 472, 180],  # BMW 530i
    [2675,  169, 220, 1350, 469, 175],  # Rover 827i
    [2849,  182, 226, 1350, 471, 176],  # Renault 25
    [1998,  115, 195, 1100, 425, 173],  # Opel
    [1998,  125, 190, 1150, 450, 176],  # Peugeot 405 break
    [1993,  112, 195, 1190, 451, 172],  # Ford Sierra
    [2494,  171, 189, 1390, 443, 169],  # BMW 325i
    [2309,  136, 198, 1290, 439, 168],  # Audi 90 Q
    [1998,  115, 188, 1290, 444, 175],  # Ford Scorpio
    [1721,   90, 180, 1080, 438, 170],  # Renault 21
    [1597,   57, 130, 1050, 399, 162],  # Nissan Vannette
    [1913,   82, 145, 1450, 457, 184],  # VW Caravelle
    [1117,   52, 148,  740, 380, 156],  # Ford Fiesta
    [1116,   58, 154,  710, 364, 155],  # Fiat Uno
    [1595,  110, 186,  950, 403, 171],  # VW Golf
    [1294,   95, 184,  840, 370, 157],  # Peugeot 205 R
    [1193,   93, 185,  900, 381, 162],  # Seat Ibiza SXI
    [1294,   95, 184,  750, 350, 160],  # Citroen AX S
], dtype=int)

n, m = X.shape

# -----------------------
# 2. Center and Standardize the Data
# -----------------------
means = np.mean(X, axis=0)
stds  = np.std(X, axis=0, ddof=1)
Y = X - means               # Center the data
D1_s = np.diag(1 / stds)     # Inverse standard deviations matrix
Z = Y @ D1_s                # Standardized data

# -----------------------
# 3. Compute the Correlation Matrix and Perform Eigen-Decomposition
# -----------------------
D = np.eye(n) * (1/(n-1))    # D matrix for unbiased covariance

# S = Y.T @ D @ Y             # Covariance Matrix
# R = D1_s @ S @ D1_s         # Correlation matrix from covariance matrix
R = Z.T @ D @ Z             # Correlation matrix from standarized data and weights

eigenvalues, eigenvectors = np.linalg.eig(R)

# Sort eigenvalues and eigenvectors in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# -----------------------
# 4. Compute Explained Variance
# -----------------------
variance_explained = (eigenvalues / np.sum(eigenvalues)) * 100
cumulative_variance = np.cumsum(variance_explained)

# -----------------------
# 5. Compute the Principal Components (Transformed Data)
# -----------------------
U = eigenvectors
C = Z @ U   # Transformed data: projection on principal components

# -----------------------
# 6. Create a DataFrame for the Variance Table
# -----------------------
components = [f'λ_{i+1}' for i in range(len(eigenvalues))]
df = pd.DataFrame({
    'Component': components,
    'Explained Variance': np.round(variance_explained, 6),
    'Cumulative Explained Variance': np.round(cumulative_variance, 6)
})
df.set_index('Component', inplace=True)

# -----------------------
# 7. Plot the Transformed Data and the Bar Plot from the DataFrame
# -----------------------
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Left subplot: Scatter plot of transformed data on first two principal components
ax[0].scatter(C[:, 0], C[:, 1], color='red', marker='^')
ax[0].set_title("Transformed Data (PC1 vs PC2)")
ax[0].set_xlabel("Principal Component 1")
ax[0].set_ylabel("Principal Component 2")
ax[0].grid(True)

# Right subplot: Bar plot from the DataFrame
# Plot the individual explained variance as bars
df['Explained Variance'].plot(kind='bar', ax=ax[1], color='skyblue', width=0.7, position=0, label='Explained Variance')

# Overlay the cumulative explained variance as a line plot
ax[1].plot(df.index, df['Cumulative Explained Variance'], marker='o', color='red', label='Cumulative Explained Variance')

ax[1].set_title("Explained Variance by Principal Components")
ax[1].set_xlabel("Principal Components")
ax[1].set_ylabel("Explained Variance (%)")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()
