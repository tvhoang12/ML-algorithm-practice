import numpy as np
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt

# gán nhãn cho các điểm dựa trên tâm cụm hiện tại
def kmeans_assign_labels(X, centroids):
    D = cdist(X, centroids)**2
    return np.argmin(D, axis=1)

# cập nhật tâm cụm mới dựa trên nhãn hiện tại
def kmeans_update_centroids(X, labels, K):
    centroids = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]
        if Xk.shape[0] > 0:
            centroids[k, :] = np.mean(Xk, axis=0)
    return centroids

def has_converged(centroids, new_centroids, tol=1e-4):
    return np.linalg.norm(centroids - new_centroids)**2 < tol

def kmeans_init_centroids(X, K):
    # Chọn K điểm ngẫu nhiên từ dữ liệu làm tâm cụm ban đầu
    return X[np.random.choice(X.shape[0], K, replace=False)]

def kmeans(X, K, max_iters=100, tol=1e-4):
    #Thực hiện thuật toán K-means.
    centroids = kmeans_init_centroids(X, K)
    labels = np.zeros(X.shape[0])
    it = 0

    while it < max_iters:
        it += 1
        new_labels = kmeans_assign_labels(X, centroids)
        new_centroids = kmeans_update_centroids(X, new_labels, K)

        if has_converged(centroids, new_centroids, tol) or \
           np.array_equal(new_labels, labels): 
            labels = new_labels
            centroids = new_centroids 
            break 

        labels = new_labels
        centroids = new_centroids

    return centroids, labels, it

np.random.seed(42)
X0 = np.random.multivariate_normal([2, 2], [[1, 0], [0, 1]], 50)
X1 = np.random.multivariate_normal([6, 6], [[1, 0], [0, 1]], 50)
X_example = np.vstack((X0, X1))
K_example = 2

# Chạy K-means
final_centroids, final_labels, iterations = kmeans(X_example, K_example)

print("Số vòng lặp:", iterations)
print("Tâm cụm cuối cùng:\n", final_centroids)

plt.scatter(X_example[final_labels == 0, 0], X_example[final_labels == 0, 1], label='Cụm 0')
plt.scatter(X_example[final_labels == 1, 0], X_example[final_labels == 1, 1], label='Cụm 1')
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], marker='X', s=100, c='red', label='Tâm cụm')
plt.xlabel("Chiều 1")
plt.ylabel("Chiều 2")
plt.title("Kết quả K-means")
plt.legend()
plt.grid(True)
plt.show()