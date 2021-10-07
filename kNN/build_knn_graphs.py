from useful_tools import pickle_manager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    original_knn_output = pickle_manager.load_pickle("../pickles/original_knn_output.pickle.lz4")
    sklearn_knn_output = pickle_manager.load_pickle("../pickles/sklearn_knn_output.pickle.lz4")
    pycaret_knn_output = pickle_manager.load_pickle("../pickles/pycaret_knn_output.pickle.lz4")

    k_values = [i for i in range(1, 6)]

    original_knn_total = []
    original_knn_individual = []
    for tup in original_knn_output:
        original_knn_total.append(tup[0] * 100)
        original_knn_individual.append([i*100 for i in tup[1]])

    sklearn_knn_total = []
    sklearn_knn_individual = []
    for tup in sklearn_knn_output:
        sklearn_knn_total.append(tup[0] * 100)
        sklearn_knn_individual.append([i*100 for i in tup[1]])

    pycaret_knn_total = []
    pycaret_knn_individual = []
    for tup in pycaret_knn_output:
        pycaret_knn_total.append(tup[0] * 100)
        pycaret_knn_individual.append([i*100 for i in tup[1]])

    # Original KNN Accuracy histogram for k=1 to k=6
    plt.bar(k_values, original_knn_total, width=0.6)
    plt.axis([0, 6, 0, 100])
    plt.title("Accuracy of Original kNN Implementation at k=1 to k=5")
    plt.xlabel("k")
    plt.ylabel("Accuracy (%)")
    plt.yticks([100 - (10 * i) for i in range(0, 11)])
    for idx, val in enumerate(original_knn_total):
        plt.text(k_values[idx] - 0.25, val + 3, str(round(val, 2)))
    plt.savefig("../figures/original_kNN_bar.png", format="png")
    plt.show()

    # sklearn KNN Accuracy histogram for k=1 to k=6
    plt.bar(k_values, sklearn_knn_total, width=0.6)
    plt.axis([0, 6, 0, 100])
    plt.title("Accuracy of scikit-learn kNN Implementation at k=1 to k=5")
    plt.xlabel("k")
    plt.ylabel("Accuracy (%)")
    plt.yticks([100 - (10 * i) for i in range(0, 11)])
    for idx, val in enumerate(sklearn_knn_total):
        plt.text(k_values[idx] - 0.25, val + 3, str(round(val, 2)))
    plt.savefig("../figures/sklearn_kNN_bar.png", format="png")
    plt.show()

    # PyCaret KNN Accuracy histogram for k=1 to k=6
    plt.bar(k_values, pycaret_knn_total, width=0.6)
    plt.axis([0, 6, 0, 100])
    plt.title("Accuracy of PyCaret kNN Implementation at k=1 to k=5")
    plt.xlabel("k")
    plt.ylabel("Accuracy (%)")
    plt.yticks([100 - (10 * i) for i in range(0, 11)])
    for idx, val in enumerate(pycaret_knn_total):
        plt.text(k_values[idx] - 0.25, val + 3, str(round(val, 2)))
    plt.savefig("../figures/pycaret_kNN_bar.png", format="png")
    plt.show()

    bounds = 20
    fh, axes = plt.subplots(1, 5)
    for idx, val in enumerate(original_knn_individual):
        # Val is an array of accuracies for all users at k=idx+1
        axes[idx].scatter(np.random.uniform(-bounds/2, bounds/2, len(val)), val, c=[[0, 0, 1]], edgecolors=[[0, 0, 0]])
        axes[idx].set_xlim(-bounds, bounds)
        axes[idx].title.set_text(f"k={idx+1}")
        axes[idx].axes.get_xaxis().set_ticks([])
        axes[idx].axes.get_xaxis().set_label_position('bottom')
        axes[idx].set_aspect(5, adjustable='box')
    # Next line from: https://stackoverflow.com/questions/16150819/common-xlabel-ylabel-for-matplotlib-subplots
    fh.text(0.04, 0.5, 'Accuracy (%)', va='center', rotation='vertical')
    fh.suptitle("Accuracy of Predicted Classes Per User for Original kNN at k=1 to k=5", y=0.98)
    plt.savefig("../figures/original_individual_plot.png", format="png")
    plt.show()

    bounds = 20
    fh, axes = plt.subplots(1, 5)
    for idx, val in enumerate(sklearn_knn_individual):
        # Val is an array of accuracies for all users at k=idx+1
        axes[idx].scatter(np.random.uniform(-bounds / 2, bounds / 2, len(val)), val, c=[[0, 0, 1]],
                          edgecolors=[[0, 0, 0]])
        axes[idx].set_xlim(-bounds, bounds)
        axes[idx].title.set_text(f"k={idx + 1}")
        axes[idx].axes.get_xaxis().set_ticks([])
        axes[idx].axes.get_xaxis().set_label_position('bottom')
        axes[idx].set_aspect(5, adjustable='box')
    # Next line from: https://stackoverflow.com/questions/16150819/common-xlabel-ylabel-for-matplotlib-subplots
    fh.text(0.04, 0.5, 'Accuracy (%)', va='center', rotation='vertical')
    fh.suptitle("Accuracy of Predicted Classes Per User for scikit-learn kNN at k=1 to k=5", y=0.98)
    plt.savefig("../figures/sklearn_individual_plot.png", format="png")
    plt.show()

    bounds = 20
    fh, axes = plt.subplots(1, 5)
    for idx, val in enumerate(pycaret_knn_individual):
        # Val is an array of accuracies for all users at k=idx+1
        axes[idx].scatter(np.random.uniform(-bounds / 2, bounds / 2, len(val)), val, c=[[0, 0, 1]],
                          edgecolors=[[0, 0, 0]])
        axes[idx].set_xlim(-bounds, bounds)
        axes[idx].title.set_text(f"k={idx + 1}")
        axes[idx].axes.get_xaxis().set_ticks([])
        axes[idx].axes.get_xaxis().set_label_position('bottom')
        axes[idx].set_aspect(5, adjustable='box')
    # Next line from: https://stackoverflow.com/questions/16150819/common-xlabel-ylabel-for-matplotlib-subplots
    fh.text(0.04, 0.5, 'Accuracy (%)', va='center', rotation='vertical')
    fh.suptitle("Accuracy of Predicted Classes Per User for PyCaret kNN at k=1 to k=5", y=0.98)
    plt.savefig("../figures/pycaret_individual_plot.png", format="png")
    plt.show()

    plt.plot(k_values, original_knn_total)
    plt.plot(k_values, sklearn_knn_total)
    plt.plot(k_values, pycaret_knn_total)
    plt.legend(["Original Accuracy", "Scikit-learn Accuracy", "PyCaret Accuracy"])
    plt.xlabel("k")
    plt.ylabel("Accuracy (%)")
    plt.xticks(k_values)
    plt.yticks([100 - (10 * i) for i in range(0, 11)])
    plt.title("Accuracy Comparison of 3 Methods at Different k Values")
    plt.savefig("../figures/all_methods_accuracy.png", format="png")
    plt.show()


if __name__ == "__main__":
    main()