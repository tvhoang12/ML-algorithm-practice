from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Tải và chuẩn bị dữ liệu
iris = load_iris()
X = iris.data  
y = iris.target  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=3, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Dữ liệu kiểm tra (thực tế): {y_test}")
print(f"Dữ liệu dự đoán (mô hình): {y_pred}")
print(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%")

# 6. Thử dự đoán một bông hoa mới
# Giả sử ta có 1 bông hoa với:
# - Chiều dài đài hoa: 5.0 cm
# - Chiều rộng đài hoa: 3.5 cm
# - Chiều dài cánh hoa: 1.5 cm
# - Chiều rộng cánh hoa: 0.2 cm
hoa_moi = [[5.0, 3.5, 1.5, 0.2]]
du_doan_hoa_moi = model.predict(hoa_moi)

print(f"Dự đoán cho hoa mới: loại {du_doan_hoa_moi[0]}")
print(f"(Tên loại hoa: {iris.target_names[du_doan_hoa_moi[0]]})")