import numpy as np
import matplotlib.pyplot as plt

def experiment_distances():
    dims = [2**i for i in range(11)]
    n_points = 100

    mean_l2, std_l2 = [], []
    mean_l1, std_l1 = [], []

    for d in dims:
        X = np.random.rand(n_points, d)
        diffs = X[:, np.newaxis, :] - X[np.newaxis, :, :]

        dists_l2 = np.sum(diffs**2, axis=2)
        dists_l1 = np.sum(np.abs(diffs), axis=2)

        triu_idx = np.triu_indices(n_points, k=1)
        l2_vals = dists_l2[triu_idx]
        l1_vals = dists_l1[triu_idx]

        mean_l2.append(np.mean(l2_vals))
        std_l2.append(np.std(l2_vals))

        mean_l1.append(np.mean(l1_vals))
        std_l1.append(np.std(l1_vals))

        # In kết quả cho từng dimension
        print(f"d={d}: L2^2 mean={mean_l2[-1]:.4f}, std={std_l2[-1]:.4f} | "
              f"L1 mean={mean_l1[-1]:.4f}, std={std_l1[-1]:.4f}")

    # Vẽ kết quả
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(dims, mean_l2, 'o-', label='Mean (ℓ2 squared)')
    plt.plot(dims, std_l2, 's-', label='Std (ℓ2 squared)')
    plt.xscale('log', base=2)
    plt.xlabel("Dimension d (log scale)")
    plt.ylabel("Value")
    plt.title("ℓ2 squared distance")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(dims, mean_l1, 'o-', label='Mean (ℓ1)')
    plt.plot(dims, std_l1, 's-', label='Std (ℓ1)')
    plt.xscale('log', base=2)
    plt.xlabel("Dimension d (log scale)")
    plt.ylabel("Value")
    plt.title("ℓ1 distance")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    experiment_distances()
