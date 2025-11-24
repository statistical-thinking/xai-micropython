######################
### XAI-MICROPYTON ###
######################

#############################
# 1. Statistical Basics - I #
#############################

# Girth (x) of Black Cherry Trees
x = [8.3, 8.6, 8.8, 10.5, 10.7, 10.8, 11, 11, 11.1, 11.2, 11.3, 
     11.4, 11.4, 11.7, 12, 12.9, 12.9, 13.3, 13.7, 13.8, 14, 
     14.2, 14.5, 16, 16.3, 17.3, 17.5, 17.9, 18, 18, 20.6]

# Mean
def mean(data):
    return sum(data) / len(data)

# Variance
def variance(data):
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / (len(data) - 1)

# Standard Deviation
def std_dev(data):
    return variance(data) ** 0.5

# Application Examples
print("Mean", mean(x))
print("Variance", variance(x))
print("Standard Deviation", std_dev(x))

##############################
# 2. Statistical Basics - II #
##############################

# Girth (x) and Volume (y) of Black Cherry Trees
x = [8.3, 8.6, 8.8, 10.5, 10.7, 10.8, 11, 11, 11.1, 11.2, 11.3, 
     11.4, 11.4, 11.7, 12, 12.9, 12.9, 13.3, 13.7, 13.8, 14, 
     14.2, 14.5, 16, 16.3, 17.3, 17.5, 17.9, 18, 18, 20.6]

y = [10.3, 10.3, 10.2, 16.4, 18.8, 19.7, 15.6, 18.2, 22.6, 19.9,
     24.2, 21, 21.4, 21.3, 19.1, 22.2, 33.8, 27.4, 25.7, 24.9,
     34.5, 31.7, 36.3, 38.3, 42.6, 55.4, 55.7, 58.3, 51.5, 51, 77]

# Covariance
def covariance(x, y):
    mx = mean(x)
    my = mean(y)
    return sum((x[i] - mx) * (y[i] - my) for i in range(len(x))) / (len(x) - 1)

# Correlation
def correlation(x, y):
    return covariance(x, y) / (std_dev(x) * std_dev(y))

# Simple Linear Regression
def linear_regression(x, y):
    b = covariance(x, y) / variance(x)
    a = mean(y) - b * mean(x)
    return a, b

# Predict Function
def predict(x_new, a, b):
    return a + b * x_new

# Residuals
def residuals(x, y, a, b):
    return [y[i] - (a + b * x[i]) for i in range(len(x))]

# Coefficient of Determination
def r_squared(x, y, a, b):
    y_mean = mean(y)
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((y[i] - (a + b * x[i])) ** 2 for i in range(len(y)))
    return 1 - ss_res / ss_tot

# Apllication Examples
print("Covariance:", covariance(x, y))
print("Correlation:", correlation(x, y))

a, b = linear_regression(x, y)
print("\nSingle Linear Regression: y = {:.2f} + {:.2f} * x".format(a, b))
print("Predictions for x = 11.4:", predict(11.4, a, b))
print("\nResiduals:", residuals(x, y, a, b))
print("\nCoefficient of Determination:", r_squared(x, y, a, b))

########################################
# 3. Machine Learning - I (Regression) #
########################################

# Girth (x1), Height (x2) and Volume (y) of Black Cherry Trees 
X = [[8.3, 70], [8.6, 65], [8.8, 63], [10.5, 72], [10.7, 81], [10.8, 83], [11, 66], [11, 75], [11.1, 80], [11.2, 75],
    [11.3, 79], [11.4, 76], [11.4, 76], [11.7, 69], [12, 75], [12.9, 74], [12.9, 85], [13.3, 86], [13.7, 71], [13.8, 64],
    [14, 78], [14.2, 80], [14.5, 74], [16, 72], [16.3, 77], [17.3, 81], [17.5, 82], [17.9, 80], [18, 80], [18, 80], [20.6, 87]]

y = [10.3, 10.3, 10.2, 16.4, 18.8, 19.7, 15.6, 18.2, 22.6, 19.9,
     24.2, 21, 21.4, 21.3, 19.1, 22.2, 33.8, 27.4, 25.7, 24.9,
     34.5, 31.7, 36.3, 38.3, 42.6, 55.4, 55.7, 58.3, 51.5, 51, 77]

# Mathematical Basics - Matrix Inversion
def invert_matrix(matrix):
    n = len(matrix)
    identity = [[float(i == j) for j in range(n)] for i in range(n)]
    m = [row[:] for row in matrix]

    for i in range(n):
        max_row = i
        max_val = abs(m[i][i])
        for k in range(i + 1, n):
            if abs(m[k][i]) > max_val:
                max_val = abs(m[k][i])
                max_row = k

        if max_val == 0:
            raise ValueError("Matrix is not invertible!")

        if max_row != i:
            m[i], m[max_row] = m[max_row], m[i]
            identity[i], identity[max_row] = identity[max_row], identity[i]

        factor = m[i][i]
        for j in range(n):
            m[i][j] /= factor
            identity[i][j] /= factor

        for k in range(n):
            if k != i:
                factor = m[k][i]
                for j in range(n):
                    m[k][j] -= factor * m[i][j]
                    identity[k][j] -= factor * identity[i][j]

    return identity

# Mathematical Basics - Matrix Transposition
def transpose(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

# Mathematical Basics - Matrix Multiplication
def matmul(A, B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            val = sum(A[i][k] * B[k][j] for k in range(len(B)))
            row.append(val)
        result.append(row)
    return result

 # Multiple Linear Regression
def multivariate_regression(X_raw, y):
    X = [[1] + row for row in X_raw]
    y_vec = [[val] for val in y]
    
    XT = transpose(X)
    XTX = matmul(XT, X)
    XTX_inv = invert_matrix(XTX)
    XTy = matmul(XT, y_vec)
    
    beta = matmul(XTX_inv, XTy)
    return [b[0] for b in beta]

# Predict Function
def predict_multi(X_raw, beta):
    X = [[1] + row for row in X_raw]
    return [sum(b * x for b, x in zip(beta, row)) for row in X]

# Residuals
def residuals_multi(X_raw, y, beta):
    y_pred = predict_multi(X_raw, beta)
    return [yi - y_hat for yi, y_hat in zip(y, y_pred)]

# Coefficient of Determination
def r_squared_multi(X_raw, y, beta):
    y_pred = predict_multi(X_raw, beta)
    y_mean = sum(y) / len(y)
    
    ss_tot = sum((yi - y_mean) ** 2 for yi in y)
    ss_res = sum((yi - y_hat) ** 2 for yi, y_hat in zip(y, y_pred))
    
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0

# Application Example
beta = multivariate_regression(X, y)
print("Coefficients:")
for i, b in enumerate(beta):
    if i == 0:
        print("a =", b)
    else:
        print("b{} = {}".format(i, b))

x_case_13 = [X[12]]
y_pred_13 = predict_multi(x_case_13, beta)[0]
print("\nPredictions for x1 = 11.4 and x2 = 76:", y_pred_13)

residuals = residuals_multi(X, y, beta)
print("\nResiduals:", residuals)

r2 = r_squared_multi(X, y, beta)
print("\nCoefficient of Determination:", r2)

#############################################
# 4. Machine Learning - II (Classification) #
#############################################

# Girth (x1), Height (x2) and Binary Volume (y) of Black Cherry Trees
X = [[8.3, 70], [8.6, 65], [8.8, 63], [10.5, 72], [10.7, 81], [10.8, 83], [11, 66], [11, 75], [11.1, 80], [11.2, 75],
    [11.3, 79], [11.4, 76], [11.4, 76], [11.7, 69], [12, 75], [12.9, 74], [12.9, 85], [13.3, 86], [13.7, 71], [13.8, 64],
    [14, 78], [14.2, 80], [14.5, 74], [16, 72], [16.3, 77], [17.3, 81], [17.5, 82], [17.9, 80], [18, 80], [18, 80], [20.6, 87]]


y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Mathematical Basics - Sigmoid Function
def sigmoid(z):
    return 1 / (1 + pow(2.71828, -z))

# Mathematical Basics - Log Function
def log(x):
    n = 1000.0
    return n * ((x**(1/n)) - 1)

# Mathematical Basics - Prediction of Probability
def predict_proba(x_input, weights, bias):
    z = bias
    for i in range(len(x_input)):
        z += weights[i] * x_input[i]
    return sigmoid(z)

# Multivariate Logistic Regression via Gradient Descent
def train_logistic_regression(X, y, lr=0.01, epochs=5000):
    n_samples = len(X)
    n_features = len(X[0])
    weights = [0] * n_features
    bias = 0

    for _ in range(epochs):
        grad_w = [0] * n_features
        grad_b = 0
        for i in range(n_samples):
            z = bias
            for j in range(n_features):
                z += weights[j] * X[i][j]
            p = sigmoid(z)
            error = p - y[i]
            for j in range(n_features):
                grad_w[j] += error * X[i][j]
            grad_b += error
        for j in range(n_features):
            weights[j] -= lr * grad_w[j] / n_samples
        bias -= lr * grad_b / n_samples

    return weights, bias

# Binary Prediction of Multivariate Logistic Regression
def predict(x_input, weights, bias):
    p = predict_proba(x_input, weights, bias)
    return 1 if p >= 0.5 else 0, p

# Application Examples
weights, bias = train_logistic_regression(X, y)
print("Weights:", weights)
print("Intercept:", bias)

print("\nLogits and Probabilities:")
for i in range(len(X)):
    z = bias + sum([weights[j] * X[i][j] for j in range(len(weights))])
    p = sigmoid(z)
    print("x =", X[i], "Logit =", z, "P(y=1) =", p)

classification = predict([11.4, 76], weights, bias)
print("\nPredicted Class:", classification)

##########################################
# 5. Machine Learning - III (Clustering) #
##########################################

# Girth (x1), Height (x2) and Class (y) of Black Cherry Trees and Simulated Trees
X = [
    [8.3, 70], [8.6, 65], [8.8, 63], [10.5, 72], [10.7, 81], [10.8, 83], [11.0, 66], [11.0, 75], [11.1, 80],
    [11.2, 75], [11.3, 79], [11.4, 76], [11.7, 69], [12.0, 75], [12.9, 74], [5.2, 45], [5.5, 48], [6.0, 50],
    [6.3, 46], [6.7, 49], [7.0, 51], [7.2, 47], [7.4, 52], [7.5, 50], [7.7, 46], [7.9, 53], [8.1, 49],
    [8.4, 47], [8.5, 54], [8.7, 52]
]

# y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Mathematical Basics - Euclidean Distance
def euclidean_distance(p1, p2):
    return sum((p1[i] - p2[i])**2 for i in range(len(p1))) ** 0.5

# Initializing Centroids Function
def initialize_centroids(X, k):
    return [X[i][:] for i in range(k)]

# Assigning Clusters Function
def assign_clusters(X, centroids):
    clusters = [[] for _ in centroids]
    for point in X:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        min_index = distances.index(min(distances))
        clusters[min_index].append(point)
    return clusters

# Computing Centroids Function
def compute_centroids(clusters):
    new_centroids = []
    for cluster in clusters:
        if not cluster:
            continue
        n_features = len(cluster[0])
        mean = [0] * n_features
        for point in cluster:
            for i in range(n_features):
                mean[i] += point[i]
        mean = [val / len(cluster) for val in mean]
        new_centroids.append(mean)
    return new_centroids

# Within Cluster Sum of Squares
def wcss(clusters, centroids):
    total = 0
    for i in range(len(clusters)):
        for point in clusters[i]:
            total += sum((point[j] - centroids[i][j])**2 for j in range(len(point)))
    return total

# K-Means Algorithm
def kmeans_wcss(X, k=2, max_iter=100):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iter):
        clusters = assign_clusters(X, centroids)
        new_centroids = compute_centroids(clusters)
        if new_centroids == centroids:
            break
        centroids = new_centroids

    labels = [0] * len(X)
    for cluster_index, cluster_points in enumerate(clusters):
        for point in cluster_points:
            for idx, original_point in enumerate(X):
                if original_point == point:
                    labels[idx] = cluster_index
                    break

    return wcss(clusters, centroids), labels, centroids

# K-Means Indicator
def kmeans_indicator(X, max_k=10):
    wcss_values = []
    for k in range(1, max_k + 1):
        wcss_value, _, _ = kmeans_wcss(X, k)
        wcss_values.append(wcss_value)
    return wcss_values

## Application examples
wcss_list = kmeans_indicator(X, max_k=6)

print("WCSS values for k = 1 to 6:")
for k, w in enumerate(wcss_list, 1):
    print("k =", k, "-> WCSS =", w)

wcss_value, cluster_labels, final_centroids = kmeans_wcss(X, k=2)

print("\nWCSS:", wcss_value)
print("\nCluster Labels:", cluster_labels)
print("\nCentroids:", final_centroids)

######################################################
# 6. Machine Leaning - IV (Dimensionality Reduction) #
######################################################

# Extraversion (x1:x5) and Neuroticism (x6:x10) of the Big Five Inventory
X = [
    [2, 1, 6, 5, 6, 3, 5, 2, 2, 3],
    [3, 6, 4, 2, 1, 6, 3, 2, 6, 4],
    [1, 3, 2, 5, 4, 3, 3, 4, 2, 3],
    [3, 4, 3, 6, 5, 2, 4, 2, 2, 3],
    [2, 1, 2, 5, 2, 2, 2, 2, 2, 2],
    [2, 2, 4, 6, 6, 4, 4, 4, 6, 6],
    [3, 2, 5, 5, 6, 2, 3, 3, 1, 1],
    [1, 1, 6, 6, 6, 2, 3, 1, 2, 1],
    [2, 4, 4, 2, 6, 3, 3, 5, 3, 2],
    [1, 2, 6, 5, 4, 1, 4, 2, 2, 5],
    [1, 2, 6, 5, 5, 5, 4, 4, 3, 1],
    [1, 2, 4, 5, 5, 3, 2, 4, 1, 2],
    [6, 6, 2, 1, 1, 1, 2, 1, 3, 6],
    [3, 4, 3, 2, 3, 5, 3, 4, 4, 3],
    [6, 6, 3, 2, 2, 2, 2, 2, 4, 1],
    [3, 4, 3, 3, 5, 5, 6, 5, 5, 4],
    [3, 2, 3, 6, 5, 1, 2, 1, 2, 1],
    [4, 3, 4, 4, 4, 2, 2, 3, 3, 3],
    [3, 3, 2, 5, 4, 2, 3, 1, 3, 2],
    [6, 4, 4, 4, 3, 2, 2, 3, 4, 5]
]

# Mean Center Function
def mean_center(X):
    cols = len(X[0])
    rows = len(X)
    means = [sum(X[i][j] for i in range(rows)) / rows for j in range(cols)]
    centered = [[X[i][j] - means[j] for j in range(cols)] for i in range(rows)]
    return centered, means

# Correlation Matrix
def correlation_matrix(X):
    rows = len(X)
    cols = len(X[0])
    corr = [[0]*cols for _ in range(cols)]

    for i in range(cols):
        for j in range(cols):
            xi = [row[i] for row in X]
            xj = [row[j] for row in X]
            num = sum(xi[k] * xj[k] for k in range(rows))
            denom_i = sum(xi[k]**2 for k in range(rows)) ** 0.5
            denom_j = sum(xj[k]**2 for k in range(rows)) ** 0.5
            corr[i][j] = num / (denom_i * denom_j)
    return corr

# Power Iteration Function 
def power_iteration(A, num_vectors=2, iterations=100):
    n = len(A)
    eigenvectors = []
    eigenvalues = []

    for _ in range(num_vectors):
        b = [1.0]*n
        for _ in range(iterations):
            # Multiply A * b
            Ab = [sum(A[i][j] * b[j] for j in range(n)) for i in range(n)]
            norm = sum(x**2 for x in Ab) ** 0.5
            b = [x / norm for x in Ab]
        # Rayleigh quotient for eigenvalue
        Ab = [sum(A[i][j] * b[j] for j in range(n)) for i in range(n)]
        eigval = sum(b[i] * Ab[i] for i in range(n))
        eigenvalues.append(eigval)
        eigenvectors.append(b)

        # Deflation
        for i in range(n):
            for j in range(n):
                A[i][j] -= eigval * b[i] * b[j]

    return eigenvalues, eigenvectors

# Factor Loadings Function
def factor_loadings(corr_matrix, eigenvalues, eigenvectors):
    loadings = []
    for i in range(len(corr_matrix)):
        row = []
        for j in range(len(eigenvectors)):
            loading = eigenvectors[j][i] * (eigenvalues[j] ** 0.5)
            row.append(loading)
        loadings.append(row)
    return loadings

# Application examples
X_centered, means = mean_center(X)
R = correlation_matrix(X_centered)
eigvals, eigvecs = power_iteration([row[:] for row in R], num_vectors=2)
loadings = factor_loadings(R, eigvals, eigvecs)

print("Correlation matrix:")
for row in R:
    print(["{0:.2f}".format(x) for x in row])

print("\nEigenvalues:")
for i, val in enumerate(eigvals):
    print("Factor", i+1, ":", round(val, 3))

print("\nFactor Loadings:")
for i, row in enumerate(loadings):
    print("V" + str(i+1), ":", ["{0:.2f}".format(x) for x in row])

########################
# 7. Deep Learning - I #
########################

# Standardized independent variables (X) and dichotomized dependent variable (y)
X = [[ 0.81575475, -0.21746808, -0.12904165, -0.65303909],
         [ 0.05761837,  1.59476592,  0.84485761,  1.71304456],
         [ 0.96738203,  0.68864892, -0.00730424, -0.41643072],
         [ 2.02877297,  0.38660992,  2.06223168,  1.00321947],
         [ 1.42226386,  0.99068792,  1.33180724,  0.29339437],
         [ 0.81575475,  0.99068792,  1.21006983,  1.4764362 ],
         [-1.00377258,  0.38660992, -0.49425387, -0.41643072],
         [ 0.05761837, -0.51950708, -0.00730424,  0.29339437],
         [ 0.36087292,  0.38660992,  1.08833242,  1.23982783],
         [ 0.66412748,  0.38660992,  0.35790798,  1.4764362 ],
         [ 0.05761837,  0.08457092,  0.84485761,  0.29339437],
         [-0.70051802, -0.51950708,  0.23617057,  0.53000274],
         [ 0.20924564, -0.21746808,  0.84485761,  1.00321947],
         [-0.24563619,  0.08457092, -0.25077906, -0.65303909],
         [-2.06516352, -1.42562408, -1.95510276, -1.59947255],
         [-1.15539985, -1.42562408, -1.34641572, -1.36286418],
         [ 0.05761837, -1.12358508, -0.00730424, -0.41643072],
         [ 0.20924564,  0.08457092, -0.73772869, -0.88964745],
         [-0.39726347, -0.51950708,  0.23617057, -0.17982236],
         [ 0.5125002 ,  0.08457092, -0.37251647, -0.88964745]]

y = [0,1,0,1,1,1,0,1,1,1,1,1,1,0,0,0,0,0,0,0]

# Libraries
import math

# ReLU
def relu(x):
    return [max(0, val) for val in x]

# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return [val if val >= 0 else alpha * val for val in x]

# Tanh
def tanh(x):
    return [(math.exp(val) - math.exp(-val)) / (math.exp(val) + math.exp(-val)) for val in x]

# Sigmoid
def sigmoid(x):
    return [1 / (1 + math.exp(-val)) for val in x]

# Single Neuron
def neuron(x, w, b, activation):

    tmp = zero_dim(x[0])

    for i in range(len(x)):
        tmp = add_dim(tmp, [(float(w[i]) * float(x[i][j])) for j in range(len(x[0]))])

    if activation == "sigmoid":
        yp = sigmoid([tmp[i] + b for i in range(len(tmp))])
    elif activation == "relu":
        yp = relu([tmp[i] + b for i in range(len(tmp))])
    elif activation == "leaky_relu":
        yp = relu([tmp[i] + b for i in range(len(tmp))])
    elif activation == "tanh":
        yp = tanh([tmp[i] + b for i in range(len(tmp))])
    else:
        print("Function unknown!")

    return yp

# Mathematical Basics - I
def zero_dim(x):
    z = [0 for i in range(len(x))]
    return z

# Mathematical Basics - II
def add_dim(x, y):
    z = [x[i] + y[i] for i in range(len(x))]
    return z

# Mathematical Basics - III
def zeros(rows, cols):
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.0)
    return M

# Mathematical Basics - IV
def transpose(M):
    if not isinstance(M[0], list):
        M = [M]
    rows = len(M)
    cols = len(M[0])
    MT = zeros(cols, rows)
    for i in range(rows):
        for j in range(cols):
            MT[j][i] = M[i][j]
    return MT

# Mathematical Basics - V
def print_matrix(M, decimals=3):
    for row in M:
        print([round(x, decimals) + 0 for x in row])

# Mathematical Basics - VI
def dense(nunit, x, w, b, activation):
    res = []
    for i in range(nunit):
        z = neuron(x, w[i], b[i], activation)
        res.append(z)
    return res

# Include Parameters from TensorFlow
w1 = [[-0.75323504, -0.25906014],
      [-0.46379513, -0.5019245 ],
      [ 2.1273055 ,  1.7724446 ],
      [ 1.1853403 ,  0.88468695]]
b1 = [0.53405946, 0.32578036]
w2 = [[-1.6785783,  2.0158117,  1.2769054],
      [-1.4055765,  0.6828738,  1.5902631]]
b2 = [ 1.18362  , -1.1555661, -1.0966455]
w3 = [[ 0.729278  , -1.0240695 ],
      [-0.80972326,  1.4383037 ],
      [-0.90892404,  1.6760625 ]]
b3 = [0.10695826, 0.01635581]
w4 = [[-0.2019448],
      [ 1.5772797]]
b4 = [-1.2177287]

# Transpose
w1 = transpose(w1)
w2 = transpose(w2)
w3 = transpose(w3)
w4 = transpose(w4)

# Neural Network Architecture
yout1 = dense(2, transpose(X), w1, b1, 'relu') # input layer (2 neurons)
yout2 = dense(3, yout1, w2, b2, 'sigmoid') # hidden layer (3 neurons)
yout3 = dense(2, yout2, w3, b3, 'relu') # hidden layer (2 neurons)
ypred = dense(1, yout3, w4, b4,'sigmoid') # output layer (1 neuron)
print(ypred)

# Confusion Matrix Basics
def classification_report(y, ypred):
    TP = TN = FP = FN = 0
    for true, pred in zip(y, ypred):
        if true == pred:
            if true == 1:
                TP += 1
            else:
                TN += 1
        else:
            if true == 1:
                FN += 1
            else:
                FP += 1
    accuracy = (TP + TN) / len(y)
    print("Accuracy: {:.3f}".format(accuracy))
    print("Confusion Matrix:")
    print("TN: {}, FP: {}".format(TN, FP))
    print("FN: {}, TP: {}".format(FN, TP))

# Confusion Matrix
ypred_class = [1 if i > 0.5 else 0 for i in ypred[0]]
print(ypred_class)
print(classification_report(y, ypred_class))

#########################
# 8. Deep Learning - II #
#########################

# Standardized independent variables (X) and dichotomized dependent variable (y)
X = [[ 0.81575475, -0.21746808, -0.12904165, -0.65303909],
         [ 0.05761837,  1.59476592,  0.84485761,  1.71304456],
         [ 0.96738203,  0.68864892, -0.00730424, -0.41643072],
         [ 2.02877297,  0.38660992,  2.06223168,  1.00321947],
         [ 1.42226386,  0.99068792,  1.33180724,  0.29339437],
         [ 0.81575475,  0.99068792,  1.21006983,  1.4764362 ],
         [-1.00377258,  0.38660992, -0.49425387, -0.41643072],
         [ 0.05761837, -0.51950708, -0.00730424,  0.29339437],
         [ 0.36087292,  0.38660992,  1.08833242,  1.23982783],
         [ 0.66412748,  0.38660992,  0.35790798,  1.4764362 ],
         [ 0.05761837,  0.08457092,  0.84485761,  0.29339437],
         [-0.70051802, -0.51950708,  0.23617057,  0.53000274],
         [ 0.20924564, -0.21746808,  0.84485761,  1.00321947],
         [-0.24563619,  0.08457092, -0.25077906, -0.65303909],
         [-2.06516352, -1.42562408, -1.95510276, -1.59947255],
         [-1.15539985, -1.42562408, -1.34641572, -1.36286418],
         [ 0.05761837, -1.12358508, -0.00730424, -0.41643072],
         [ 0.20924564,  0.08457092, -0.73772869, -0.88964745],
         [-0.39726347, -0.51950708,  0.23617057, -0.17982236],
         [ 0.5125002 ,  0.08457092, -0.37251647, -0.88964745]]

y = [0,1,0,1,1,1,0,1,1,1,1,1,1,0,0,0,0,0,0,0]

# Libraries
import random
import math

# Sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Derivate of Sigmoid
def sigmoid_derivative(output):
    return output * (1 - output)

# ReLU
def relu(x):
    return max(0, x)

# Derivate of ReLU
def relu_derivative(output):
    return 1 if output > 0 else 0

# Function for Initializing Weights and Biases
def init_layer(input_size, output_size):
    weights = [[random.uniform(-0.5, 0.5) for _ in range(input_size)] for _ in range(output_size)]
    biases = [random.uniform(-0.5, 0.5) for _ in range(output_size)]
    return weights, biases

# Forward Propagation
def dense_forward(inputs, weights, biases, activation='relu'):
    outputs = []
    pre_activations = []
    for w, b in zip(weights, biases):
        z = sum(i*w_ij for i, w_ij in zip(inputs, w)) + b
        pre_activations.append(z)
        if activation == 'sigmoid':
            outputs.append(sigmoid(z))
        elif activation == 'relu':
            outputs.append(relu(z))
        else:
            raise Exception("Unknown activation")
    return outputs, pre_activations

# Backward Propagation
def dense_backward(inputs, grad_outputs, outputs, pre_activations, weights, biases, activation='relu', lr=0.01):
    input_grads = [0.0 for _ in range(len(inputs))]
    for j in range(len(weights)):
        if activation == 'sigmoid':
            delta = grad_outputs[j] * sigmoid_derivative(outputs[j])
        elif activation == 'relu':
            delta = grad_outputs[j] * relu_derivative(pre_activations[j])
        else:
            raise Exception("Unknown activation")
        for i in range(len(inputs)):
            input_grads[i] += weights[j][i] * delta
            weights[j][i] -= lr * delta * inputs[i]
        biases[j] -= lr * delta
    return input_grads

# Loss Function
def binary_cross_entropy(predicted, target):
    epsilon = 1e-7
    return - (target * math.log(predicted + epsilon) + (1 - target) * math.log(1 - predicted + epsilon))

def binary_cross_entropy_derivative(predicted, target):
    epsilon = 1e-7
    return -(target / (predicted + epsilon)) + (1 - target) / (1 - predicted + epsilon)

# Initialize Weights and Biases
w1, b1 = init_layer(4, 3)
w2, b2 = init_layer(3, 1)

# Epochs and Learning Rate for Training
epochs = 100
lr = 0.05

for epoch in range(epochs):
    total_loss = 0
    for xi, yi in zip(X, y):
        # Forward pass
        out1, pre1 = dense_forward(xi, w1, b1, 'relu')
        out2, pre2 = dense_forward(out1, w2, b2, 'sigmoid')
        loss = binary_cross_entropy(out2[0], yi)
        total_loss += loss

        # Backward pass
        dL_dout2 = [binary_cross_entropy_derivative(out2[0], yi)]
        dL_dout1 = dense_backward(out1, dL_dout2, out2, pre2, w2, b2, 'sigmoid', lr)
        _ = dense_backward(xi, dL_dout1, out1, pre1, w1, b1, 'relu', lr)

    if epoch % 10 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Predict Function
def predict(x):
    out1, _ = dense_forward(x, w1, b1, 'relu')
    out2, _ = dense_forward(out1, w2, b2, 'sigmoid')
    return 1 if out2[0] > 0.5 else 0

ypred = [predict(xi) for xi in X]

# Confusion Matrix Basics
def classification_report(ytrue, ypred):
    TP = TN = FP = FN = 0
    for true, pred in zip(ytrue, ypred):
        if true == pred:
            if true == 1:
                TP += 1
            else:
                TN += 1
        else:
            if true == 1:
                FN += 1
            else:
                FP += 1
    accuracy = (TP + TN) / len(ytrue)
    print("Accuracy: {:.3f}".format(accuracy))
    print("Confusion Matrix:")
    print("TN: {}, FP: {}".format(TN, FP))
    print("FN: {}, TP: {}".format(FN, TP))

# Generate predictions
ypred = [predict(xi) for xi in X]

# Show classification metrics
classification_report(y, ypred)
