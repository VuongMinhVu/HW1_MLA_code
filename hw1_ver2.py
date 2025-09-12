import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def load_data(fake_path="clean_fake.txt", real_path="clean_real.txt", random_state=42):
    """
    Load dữ liệu từ file txt, vector hóa bằng CountVectorizer,
    và chia dữ liệu thành train/val/test theo tỉ lệ 70/15/15.
    """
    # Đọc dữ liệu từ file
    with open(fake_path, "r", encoding="utf-8") as f:
        fake_lines = f.readlines()
    with open(real_path, "r", encoding="utf-8") as f:
        real_lines = f.readlines()

    fake_lines = [line.strip() for line in fake_lines if line.strip()]
    real_lines = [line.strip() for line in real_lines if line.strip()]

    # Tạo nhãn: fake = 0, real = 1
    X = fake_lines + real_lines
    y = [0] * len(fake_lines) + [1] * len(real_lines)

    # Vector hóa
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    # Tách train (70%) và temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_vec, y, test_size=0.30, random_state=random_state, stratify=y
    )
    # Chia tiếp temp thành validation (15%) và test (15%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    return X_train, X_val, X_test, np.array(y_train), np.array(y_val), np.array(y_test), vectorizer


def select_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Huấn luyện DecisionTreeClassifier với nhiều giá trị max_depth và criterion.
    In ra accuracy trên validation set, vẽ biểu đồ accuracy theo max_depth,
    và báo cáo test accuracy cho model tốt nhất.
    """
    criteria = ["gini", "entropy", "log_loss"]
    max_depths = [2, 4, 6, 8, 10, 15, 20]

    best_val_acc = 0
    best_params = None
    results = []

    for criterion in criteria:
        val_accuracies = []
        for depth in max_depths:
            clf = DecisionTreeClassifier(criterion=criterion, max_depth=depth, random_state=42)
            clf.fit(X_train, y_train)
            y_val_pred = clf.predict(X_val)
            acc = accuracy_score(y_val, y_val_pred)
            val_accuracies.append(acc)

            results.append((criterion, depth, acc))

            # Cập nhật model tốt nhất
            if acc > best_val_acc:
                best_val_acc = acc
                best_params = (criterion, depth, clf)

        # Vẽ biểu đồ cho từng criterion
        plt.plot(max_depths, val_accuracies, marker="o", label=criterion)

    plt.xlabel("Max Depth")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs Max Depth")
    plt.legend()
    plt.grid(True)

    # Chỉ hiển thị, không lưu file
    plt.show()

    # In kết quả
    print("Validation results:")
    for criterion, depth, acc in results:
        print(f"Criterion={criterion}, Max Depth={depth}, Val Accuracy={acc:.4f}")

    # Đánh giá trên test với best model
    best_clf = best_params[2]
    y_test_pred = best_clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print("\nBest model:")
    print(f"Criterion={best_params[0]}, Max Depth={best_params[1]}")
    print(f"Validation Accuracy={best_val_acc:.4f}")
    print(f"Test Accuracy={test_acc:.4f}")

    return best_clf, results, test_acc


if __name__ == "__main__":
    # Load dữ liệu
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = load_data(
        fake_path="clean_fake.txt", real_path="clean_real.txt"
    )

    # Train và chọn model
    best_model, results, test_acc = select_model(X_train, y_train, X_val, y_val, X_test, y_test)

    print("\nDone!")
