from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict


def load_data(tsv_file):
    data = defaultdict(dict)
    with open(tsv_file) as f:
        for line in f:
            n, k, auc = line.strip().split('\t')
            data[int(n)][int(k)] = float(auc)

    x_labels = sorted(data.keys())
    y_labels = sorted(next(iter(data.values())).keys(), reverse=True)

    matrix = []
    for k in y_labels:
        new_row = []
        for n in x_labels:
            new_row.append(data[n][k] * 100)
        matrix.append(new_row)

    return matrix, x_labels, y_labels


def find_two_minima(matrix):
    min1, min2 = sorted([v for row in matrix for v in row])[:2]
    return min1, min2


def main(args):
    matrix, x_labels, y_labels = load_data(args.tsv_file)

    # min1, min2 = find_two_minima(matrix)
    # vmin = min1 if min2 - min1 < 20 else min2

    ax = sns.heatmap(
        matrix, annot=True, fmt='.1f',
        xticklabels=x_labels, yticklabels=y_labels,
        cmap=plt.cm.seismic, center=matrix[-1][0],
        robust=True
        # vmin=vmin
    )
    ax.set_ylabel('#tokens per pattern $k$')
    ax.set_xlabel('#patterns $n$')
    plt.savefig(args.plot_file, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('tsv_file')
    parser.add_argument('plot_file')
    args = parser.parse_args()

    main(args)
