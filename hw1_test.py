import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from math import log2


# ================== Load Data ==================
def load_data(fake_file="clean_fake.txt", real_file="clean_real.txt"):
    with open(fake_file, "r", encoding="utf-8") as f:
        fake_lines = f.readlines()
    with open(real_file, "r", encoding="utf-8") as f:
        real_lines = f.readlines()

    X = fake_lines + real_lines
    y = [0] * len(fake_lines) + [1] * len(real_lines)  # 0=fake, 1=real

    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_vec, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer


# ================== Ý a ==================
def dataset_and_baseline(X_train, X_val, X_test, y_train, y_val, y_test, vectorizer):
    total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    print(f"Total examples: {total}")
    print(f"Training set: {X_train.shape[0]} examples ({100*X_train.shape[0]/total:.1f}%)")
    print(f"Validation set: {X_val.shape[0]} examples ({100*X_val.shape[0]/total:.1f}%)")
    print(f"Test set: {X_test.shape[0]} examples ({100*X_test.shape[0]/total:.1f}%)")
    print(f"Number of features: {X_train.shape[1]}")

    def count_classes(y):
        fake = sum(1 for i in y if i == 0)
        real = sum(1 for i in y if i == 1)
        return fake, real

    tr_fake, tr_real = count_classes(y_train)
    val_fake, val_real = count_classes(y_val)
    te_fake, te_real = count_classes(y_test)

    print(f"\nTraining set - Fake: {tr_fake}, Real: {tr_real}")
    print(f"Validation set - Fake: {val_fake}, Real: {val_real}")
    print(f"Test set - Fake: {te_fake}, Real: {te_real}")

    # Baseline Decision Tree
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    acc_train = accuracy_score(y_train, clf.predict(X_train))
    acc_val = accuracy_score(y_val, clf.predict(X_val))
    acc_test = accuracy_score(y_test, clf.predict(X_test))

    print("\nBasic Decision Tree Results:")
    print(f"Training accuracy: {acc_train:.3f}")
    print(f"Validation accuracy: {acc_val:.3f}")
    print(f"Test accuracy: {acc_test:.3f}")


# ================== Ý b ==================
def select_model(X_train, y_train, X_val, y_val, X_test, y_test):
    criteria = ["gini", "entropy", "log_loss"]
    max_depths = [2, 4, 6, 8, 10]
    results = {}

    for crit in criteria:
        accs = []
        for d in max_depths:
            clf = DecisionTreeClassifier(criterion=crit, max_depth=d, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            accs.append(acc)
            print(f"Criterion={crit}, max_depth={d}, val_acc={acc:.4f}")
        results[crit] = accs

    plt.figure(figsize=(8, 6))
    for crit in criteria:
        plt.plot(max_depths, results[crit], marker="o", label=crit)
    plt.xlabel("max_depth")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs max_depth")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Chọn mô hình tốt nhất
    best_crit, best_depth, best_acc = None, None, 0
    for crit in criteria:
        for d, acc in zip(max_depths, results[crit]):
            if acc > best_acc:
                best_acc = acc
                best_crit = crit
                best_depth = d

    clf_best = DecisionTreeClassifier(criterion=best_crit, max_depth=best_depth, random_state=42)
    clf_best.fit(np.vstack([X_train.toarray(), X_val.toarray()]), y_train + y_val)
    y_test_pred = clf_best.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"\nBest model: criterion={best_crit}, max_depth={best_depth}, val_acc={best_acc:.4f}, test_acc={test_acc:.4f}")
    return clf_best, best_crit, best_depth


# ================== Ý c ==================
def visualize_tree(clf, vectorizer):
    feature_names = vectorizer.get_feature_names_out()
    plt.figure(figsize=(12, 6))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=["Fake", "Real"],
        filled=True,
        max_depth=2,
    )
    plt.title("First Two Layers of Decision Tree")
    plt.show()


# ================== Ý d ==================
def entropy(labels):
    total = len(labels)
    if total == 0:
        return 0
    counts = np.bincount(labels)
    probs = counts / total
    return -sum(p * log2(p) for p in probs if p > 0)


def compute_information_gain_all(X_train, y_train, vectorizer, clf, top_k=5, keywords=None):
    feature_names = vectorizer.get_feature_names_out()
    IG_scores = {}

    # Tính IG cho tất cả features
    for word, idx in vectorizer.vocabulary_.items():
        has_kw = X_train[:, idx].toarray().ravel() > 0
        left = [y_train[i] for i in range(len(y_train)) if has_kw[i]]
        right = [y_train[i] for i in range(len(y_train)) if not has_kw[i]]

        H_parent = entropy(y_train)
        H_left = entropy(left)
        H_right = entropy(right)

        IG = H_parent - (len(left)/len(y_train))*H_left - (len(right)/len(y_train))*H_right
        IG_scores[word] = IG

    # Lấy top k theo IG
    top_features = sorted(IG_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    print(" Information Gain Analysis:")
    print("="*50)
    for rank, (word, ig) in enumerate(top_features, 1):
        idx = np.where(feature_names == word)[0][0]
        importance = clf.feature_importances_[idx]
        print(f"Top feature {rank}: '{word}'")
        print(f"  Information Gain: {ig:.6f}")
        print(f"  Feature Importance: {importance:.6f}\n")

    # In top split feature của cây (root node)
    root_idx = clf.tree_.feature[0]
    if root_idx >= 0:
        root_feature = feature_names[root_idx]
        print(f"\nTop split feature (root node): '{root_feature}'\n")

    # In IG cho các từ khóa cụ thể
    if keywords:
        print("Information Gain for specific keywords:")
        print("="*50)
        for kw in keywords:
            if kw in vectorizer.vocabulary_:
                ig = IG_scores[kw]
                idx = vectorizer.vocabulary_[kw]
                importance = clf.feature_importances_[idx]
                print(f"Keyword: '{kw}'")
                print(f"  Information Gain: {ig:.6f}")
                print(f"  Feature Importance: {importance:.6f}\n")
            else:
                print(f"Keyword: '{kw}' not in vocabulary.\n")


# ================== Main ==================
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = load_data()

    # Ý a
    dataset_and_baseline(X_train, X_val, X_test, y_train, y_val, y_test, vectorizer)

    # Ý b
    clf_best, best_crit, best_depth = select_model(X_train, y_train, X_val, y_val, X_test, y_test)

    # Ý c
    visualize_tree(clf_best, vectorizer)

    # Ý d
    compute_information_gain_all(
        X_train, y_train, vectorizer, clf_best,
        top_k=5,
        keywords=["trump", "news", "media", "fake", "real"]
    )
