import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from math import log2


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


def hyperparameter_analysis(X_train, y_train, X_val, y_val):
    # max_depth
    depths = [5, 10, 15, 20]
    accs_depth = []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        accs_depth.append(accuracy_score(y_val, y_pred))

    # min_samples_split
    splits = [2, 5, 10]
    accs_split = []
    for s in splits:
        clf = DecisionTreeClassifier(min_samples_split=s, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        accs_split.append(accuracy_score(y_val, y_pred))

    # min_samples_leaf
    leafs = [1, 5, 10]
    accs_leaf = []
    for l in leafs:
        clf = DecisionTreeClassifier(min_samples_leaf=l, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        accs_leaf.append(accuracy_score(y_val, y_pred))

    # Vẽ 3 biểu đồ cột
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].bar(depths, accs_depth)
    axs[0].set_title("Max Depth vs Accuracy")
    axs[0].set_xlabel("Max Depth")
    axs[0].set_ylabel("Validation Accuracy")

    axs[1].bar(splits, accs_split)
    axs[1].set_title("Min Samples Split vs Accuracy")
    axs[1].set_xlabel("Min Samples Split")
    axs[1].set_ylabel("Validation Accuracy")

    axs[2].bar(leafs, accs_leaf)
    axs[2].set_title("Min Samples Leaf vs Accuracy")
    axs[2].set_xlabel("Min Samples Leaf")
    axs[2].set_ylabel("Validation Accuracy")

    plt.tight_layout()
    plt.show()


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

    plt.figure(figsize=(8,6))
    for crit in criteria:
        plt.plot(max_depths, results[crit], marker='o', label=crit)
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
    print(f"Best model: criterion={best_crit}, max_depth={best_depth}, val_acc={best_acc:.4f}, test_acc={test_acc:.4f}")
    return clf_best, best_crit, best_depth



def visualize_tree(clf, vectorizer):
    feature_names = vectorizer.get_feature_names_out()
    plt.figure(figsize=(12,6))
    plot_tree(clf, feature_names=feature_names, class_names=["Fake","Real"],
              filled=True, max_depth=2)
    plt.title("First Two Layers of Decision Tree")
    plt.show()


def entropy(labels):
    total = len(labels)
    if total == 0:
        return 0
    counts = np.bincount(labels)
    probs = counts / total
    return -sum(p * log2(p) for p in probs if p > 0)

def compute_information_gain(X_train, y_train, vectorizer, keyword):
    if keyword not in vectorizer.vocabulary_:
        print(f"Keyword '{keyword}' not in vocabulary.")
        return None
    idx = vectorizer.vocabulary_[keyword]
    has_kw = X_train[:, idx].toarray().ravel() > 0
    left = [y_train[i] for i in range(len(y_train)) if has_kw[i]]
    right = [y_train[i] for i in range(len(y_train)) if not has_kw[i]]

    H_parent = entropy(y_train)
    H_left = entropy(left)
    H_right = entropy(right)

    IG = H_parent - (len(left)/len(y_train))*H_left - (len(right)/len(y_train))*H_right
    print(f"Information gain for keyword '{keyword}': {IG:.4f}")
    return IG


# ========== Main ==========
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = load_data()
    hyperparameter_analysis(X_train, y_train, X_val, y_val)   # Ý a
    clf_best, best_crit, best_depth = select_model(X_train, y_train, X_val, y_val, X_test, y_test)  # Ý b
    visualize_tree(clf_best, vectorizer)   # Ý c
    for kw in ["trump", "hillary", "russia", "america"]:      # Ý d
        compute_information_gain(X_train, y_train, vectorizer, kw)
