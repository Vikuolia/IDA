import numpy as np
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import v_measure_score
from sklearn.metrics.cluster import contingency_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from sklearn.model_selection import train_test_split

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def show_clusters(model, X, y_pred):
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=45, cmap='viridis')
    centers = model.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=150, alpha=0.75)
    plt.title(f'Estimate number of clusters: {len(centers)}')
    plt.show()


def create_matrix(y_true, y_predict):
    cm = contingency_matrix(y_true, y_predict)
    sns.heatmap(cm, annot=True, square=True)
    plt.ylabel("Actual classes")
    plt.xlabel("Predicted clusters")
    plt.title("Contingency matrix")
    plt.show()


def get_scores(model, X, y_true):
    y_predict = model.predict(X)
    ch_index = calinski_harabasz_score(X, y_predict)
    db_index = davies_bouldin_score(X, y_predict)
    v_score = v_measure_score(y_true, y_predict)
    num_clusters = len(model.cluster_centers_indices_)
    return y_predict, ch_index, db_index, v_score, num_clusters


def metrics_plot(model, X, y_true):
    y_predict, ch_index, db_index, _, _ = get_scores(model, X, y_true)
    show_clusters(model, X, y_predict)
    print(f'Calinski-harabasz index: {ch_index}')
    print(f'Davies-bouldin index: {db_index}')
    create_matrix(y_true, y_predict)


def estimate_models(X, y_true, affinity, damping, preference):
    affinity_params = []
    v_scores = []
    ch_indices = []
    db_indices = []
    clusters_num = []

    for aff in affinity:
        for damp in damping:
            for pref in preference:
                try:
                    AP = AffinityPropagation(affinity=aff, damping=damp, preference=pref).fit(X)
                    _, ch_index, db_index, v_score, num_clusters = get_scores(AP, X, y_true)
                    affinity_params.append([aff, damp, pref])
                    ch_indices.append(ch_index)
                    db_indices.append(db_index)
                    v_scores.append(v_score)
                    clusters_num.append(num_clusters)

                except Exception as exc:
                    continue

    return pd.DataFrame({"params[affinity, damping, preference]": affinity_params,
                         "V_scores": v_scores,
                         "cluster_number": clusters_num,
                         "CH_scores": ch_indices,
                         "DB_scores": db_indices})


def estimate_size_quality(model, X, y_true):
    samples = [i/10 for i in range(1, 6, 1)]
    ch_scores = []
    bd_scores = []
    v_scores = []

    for i in samples:
        x_samples, _, y_samples, _ = train_test_split(X, y_true, test_size=i)
        _, ch, bd, v, _ = get_scores(model, x_samples, y_samples)
        ch_scores.append(ch)
        bd_scores.append(bd)
        v_scores.append(v)
        print('Metrics with ', i*100, '% data remove')
        metrics_plot(model, x_samples, y_samples)

    samples = [i*10 for i in range(5, 10)]
    plt.plot(samples, ch_scores)
    plt.legend(['Calinsky_Harabasz_score'])
    plt.xlabel("% of data used for training")
    plt.show()

    plt.plot(samples, bd_scores)
    plt.plot(samples, v_scores)
    plt.legend(['Davies_Bouldin_score', 'V_score'])
    plt.xlabel("% of data used for training")
    plt.show()


if __name__ == '__main__':

    # create datasets
    X_moons, y_moons = make_moons(n_samples=500, noise=0.15)
    X_blobs, y_blobs = make_blobs(n_samples=500,
                                  n_features=2,
                                  centers=4,
                                  cluster_std=1,
                                  center_box=(-10.0, 10.0),
                                  shuffle=True,
                                  random_state=1)

    # show datasets
    plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons)
    plt.show()

    plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs)
    plt.show()

    # create models:
    model_moons = AffinityPropagation().fit(X_moons)
    model_blobs = AffinityPropagation().fit(X_blobs)

    # show clusters
    print('\n* The first dataset *')
    metrics_plot(model_moons, X_moons, y_moons)
    print('\n* The second dataset *')
    metrics_plot(model_blobs, X_blobs, y_blobs)

    # change distances
    new_preference1 = -1*np.amax(euclidean_distances(X_moons))**1.2
    new_preference2 = -1*np.amax(euclidean_distances(X_blobs))**1.8

    model_moons_change_pref = AffinityPropagation(preference=new_preference1).fit(X_moons)
    model_blobs_change_pref = AffinityPropagation(preference=new_preference2).fit(X_blobs)

    print('\n* The first dataset with changed preference *')
    metrics_plot(model_moons_change_pref, X_moons, y_moons)
    print('\n* The second dataset with changed preference *')
    metrics_plot(model_blobs_change_pref, X_blobs, y_blobs)

    # estimate models
    affinity_check = ['euclidean', 'precomputed']
    damping_check = [i / 10 for i in range(5, 10)]
    preferences1 = [i for i in range(0, -14, -1)]
    preferences2 = [i for i in range(-200, -300, -1)]

    print('\n---------search for optimal parameters---------\n')

    # first dataset
    data_frame_moons = estimate_models(X_moons, y_moons, affinity_check, damping_check, preferences1).\
        set_index("params[affinity, damping, preference]")
    data_frame_moons = data_frame_moons[data_frame_moons["cluster_number"] != 0]
    print('\nMetrics of the first dataset', data_frame_moons)

    data_frame_moons_ch = data_frame_moons.sort_values(['CH_scores'], ascending=False)
    param = data_frame_moons_ch.index.values[0]
    print('\nThe best model on ch-score:\n', param)
    best = AffinityPropagation(affinity=param[0], damping=param[1], preference=param[2]).fit(X_moons)
    print(metrics_plot(best, X_moons, y_moons))

    data_frame_moons_db = data_frame_moons.sort_values(['DB_scores'])
    param = data_frame_moons_db.index.values[0]
    print('\nThe best model on db-score:\n', param)
    best_db = AffinityPropagation(affinity=param[0], damping=param[1], preference=param[2]).fit(X_moons)
    print(metrics_plot(best_db, X_moons, y_moons))

    data_frame_moons_v = data_frame_moons.sort_values(['V_scores'], ascending=False)
    param = data_frame_moons_v.index.values[0]
    print('\nThe best model on v-score:\n', param)
    best = AffinityPropagation(affinity=param[0], damping=param[1], preference=param[2]).fit(X_moons)
    print(metrics_plot(best, X_moons, y_moons))

    # second dataset
    data_frame_blobs = estimate_models(X_blobs, y_blobs, affinity_check, damping_check, preferences2).\
        set_index("params[affinity, damping, preference]")
    data_frame_blobs = data_frame_blobs[data_frame_blobs["cluster_number"] != 0]
    print('\nMetrics of the second dataset', data_frame_blobs)

    data_frame_blobs_ch = data_frame_blobs.sort_values(['CH_scores'], ascending=False)
    param = data_frame_blobs_ch.index[0]
    print('\nThe best model on ch-score:\n', param)
    best = AffinityPropagation(affinity=param[0], damping=param[1], preference=param[2]).fit(X_blobs)
    print(metrics_plot(best, X_blobs, y_blobs))

    data_frame_blobs_db = data_frame_blobs.sort_values(['DB_scores'])
    param = data_frame_blobs_db.index[0]
    print('\nThe best model on db-score:\n', param)
    best_db = AffinityPropagation(affinity=param[0], damping=param[1], preference=param[2]).fit(X_blobs)
    print(metrics_plot(best_db, X_blobs, y_blobs))

    data_frame_blobs_v = data_frame_blobs.sort_values(['V_scores'], ascending=False)
    param = data_frame_blobs_v.head().index[0]
    print('\nThe best model on v-score:\n', param)
    best = AffinityPropagation(affinity=param[0], damping=param[1], preference=param[2]).fit(X_blobs)
    print(metrics_plot(best, X_blobs, y_blobs))

    estimate_size_quality(model_moons_change_pref, X_moons, y_moons)
    estimate_size_quality(model_blobs_change_pref, X_blobs, y_blobs)














