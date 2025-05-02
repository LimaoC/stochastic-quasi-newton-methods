import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sqnm.methods.lbfgs import l_bfgs

from .classify import test_classify
from .logistic_fun import logistic_f, logistic_grad_f


def load_data(file):
    with open(file, "rb") as file_handle:
        raw_data = np.loadtxt(file_handle, delimiter=",", skiprows=0)

    raw_train = raw_data[:, 0:-1]
    raw_labels = raw_data[:, -1]

    n = raw_labels.size
    test_size = np.ceil(n / 5).astype(int)
    test_index = np.random.choice(n, test_size, replace=False)
    train_index = np.setdiff1d(np.arange(len(raw_labels)), test_index)

    A_train = raw_train[train_index, :]
    b_train = raw_labels[train_index].reshape(-1, 1)
    A_test = raw_train[test_index, :]
    b_test = raw_labels[test_index].reshape(-1, 1)
    b_test = np.append(b_test, 1 - b_test, axis=1)

    return A_train, b_train, A_test, b_test


def main():
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)

    data_dir = Path(__file__).parent / "data"
    A_train, b_train, A_test, b_test = load_data(data_dir / "spambase/spambase.data")
    logger.info(f"A_train = {A_train.shape}")
    logger.info(f"b_train = {b_train.shape}")
    logger.info(f"A_test  = {A_test.shape}")
    logger.info(f"b_test  = {b_test.shape}")

    # Logistic regression function
    f = logistic_f(A_train, b_train)
    grad_f = logistic_grad_f(A_train, b_train)

    _, d = A_train.shape
    x0 = np.zeros(d)
    iterates = []

    def callback(k, x_k, f_x_k, norm_grad_f_x_k):
        accuracy = test_classify(A_test, b_test, x_k)
        iterates.append((norm_grad_f_x_k, accuracy))
        if k % 100 == 0:
            logger.info(
                f"k = {k:4}, grad norm = {norm_grad_f_x_k:12.4f}, acc = {accuracy:.3f}"
            )

    start = time.time()
    l_bfgs(f, grad_f, x0, callback=callback)
    logger.info(f"Took {time.time() - start:.3f} seconds")

    plt.subplot(312)
    plt.title("Gradient norm vs. iterations")
    plt.loglog(range(len(iterates)), [it[0] for it in iterates], label="L-BFGS")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Gradient norm")
    plt.show()


if __name__ == "__main__":
    main()
