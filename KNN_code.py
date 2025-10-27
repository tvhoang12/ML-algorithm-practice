import numpy as np


def calculate_squared_distances(x_new, X_train):
  x_new = np.asarray(x_new)
  X_train = np.asarray(X_train)

  # norms and dot product trick: ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
  x_new_norm_sq = np.sum(x_new ** 2)
  X_train_norm_sq = np.sum(X_train ** 2, axis=1)
  dot_product = X_train.dot(x_new)

  distances_sq = x_new_norm_sq + X_train_norm_sq - 2 * dot_product

  # Numerical safety: clipped to >= 0
  distances_sq = np.maximum(distances_sq, 0.0)

  return distances_sq


def get_k_nearest_indices(distances_sq, k):
  distances_sq = np.asarray(distances_sq)
  if distances_sq.ndim != 1:
    raise ValueError("distances_sq must be a 1-D array of distances")

  n = distances_sq.shape[0]
  if not isinstance(k, int):
    try:
      k = int(k)
    except Exception:
      raise ValueError("k must be an integer")

  if k <= 0:
    raise ValueError("k must be > 0")

  if k > n:
    # clamp k to number of samples
    k = n

  # argsort returns indices that would sort the array ascending
  sorted_idx = np.argsort(distances_sq)
  return sorted_idx[:k]


def predict_label_knn(x_new, X_train, y_train, k=3):
  distances_sq = calculate_squared_distances(x_new, X_train)
  neighbor_idx = get_k_nearest_indices(distances_sq, k)

  neighbor_labels = np.asarray(y_train)[neighbor_idx]

  # majority vote: use bincount (assumes non-negative integer labels)
  counts = np.bincount(neighbor_labels)
  predicted = np.argmax(counts)

  return predicted, neighbor_idx


if __name__ == "__main__":
  # Small example / sanity check
  X_train_example = np.array([[1, 2], [3, 4], [5, 1]])
  y_train_example = np.array([0, 1, 1])
  x_new_example = np.array([2, 3])

  distances = calculate_squared_distances(x_new_example, X_train_example)
  print("Bình phương khoảng cách tới các điểm huấn luyện:", distances)

  # lấy chỉ số các điểm theo thứ tự khoảng cách tăng dần
  sorted_indices = np.argsort(distances)
  print("Chỉ số các điểm huấn luyện theo thứ tự khoảng cách tăng dần:", sorted_indices)

  # ví dụ chọn k = 2
  k = 2
  nearest_idx = get_k_nearest_indices(distances, k)
  print(f"{k} điểm gần nhất có chỉ số:", nearest_idx)
  print("Nhãn của các láng giềng:", y_train_example[nearest_idx])

  pred_label, neigh_idx = predict_label_knn(x_new_example, X_train_example, y_train_example, k=2)
  print("Dự đoán nhãn (k=2):", pred_label)
