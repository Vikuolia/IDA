from sklearn.datasets import make_moons
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics as mt
from mlxtend.plotting import plot_decision_regions
import scikitplot.metrics as skplt
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def show_moons(X, y):
    df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
    colors = {0: 'red', 1: 'midnightblue'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label='Class: '+str(key), color=colors[key])
    plt.show()


def show_digits(digits):
    fig = plt.figure(figsize=(6, 6))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(64):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
        ax.text(0, 7, str(digits.target[i]))
    plt.show()


def create_models(X_train, y_train):
    simple_lr = LogisticRegression(penalty='none').fit(X_train, y_train)
    simple_lr_regularized = LogisticRegression().fit(X_train, y_train)

    multinomial_lr = LogisticRegression(penalty='none', multi_class='multinomial', solver='lbfgs', max_iter=1000)\
        .fit(X_train, y_train)
    multinomial_lr_regularized = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)\
        .fit(X_train, y_train)

    return simple_lr, multinomial_lr, simple_lr_regularized, multinomial_lr_regularized


def create_matrix(curr_model, x, y_test, y_predict):
    cm = mt.confusion_matrix(y_test, y_predict)
    score = curr_model.score(x, y_test)
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot=True, fmt=".3f", square=True)
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.title(f"Accuracy Score:  {score}", size=15)
    plt.show()


def estimate(curr_model, x, y_true, num_classes):
    y_predict = curr_model.predict(x)
    create_matrix(curr_model, x, y_true, y_predict)
    print('Precision-score: ', mt.precision_score(y_true, y_predict, average='micro'))
    print('Recall-score: ', mt.recall_score(y_true, y_predict, average='micro'))
    print('F1-score: ', mt.f1_score(y_true, y_predict, average='micro'))

    if num_classes == 2:
        pr_curve = mt.plot_precision_recall_curve(curr_model, x, y_true)
        pr_curve.ax_.set_title('Precision-Recall curve')
        plt.show()

        roc_curve_moons = mt.plot_roc_curve(curr_model, x, y_true)
        roc_curve_moons.ax_.set_title('ROC curve')
        plt.show()

    y_probs = curr_model.predict_proba(x)

    if num_classes > 2:
        skplt.plot_precision_recall(y_true, y_probs)
        plt.show()

        skplt.plot_roc(y_true, y_probs)
        plt.show()


def estimate_size_quality(model, x, y):
    train_test_ratio = [i/10 for i in range(1, 10, 1)]
    f1_scores = []

    for i in train_test_ratio:
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=i)
        model.fit(x_train, y_train)
        f1_scores.append(mt.f1_score(y_test, model.predict(x_test), average='micro'))

    train_test_ratio = [i*10 for i in range(1, 10, 1)]
    plt.plot(train_test_ratio, f1_scores)
    plt.ylabel("Model's f1 score")
    plt.show()


if __name__ == '__main__':

    # create datasets
    X_moons, y_moons = make_moons(n_samples=1500, noise=0.15)
    digits_ = load_digits()

    # show datasets
    show_moons(X_moons, y_moons)
    show_digits(digits_)

    # data preprocessing
    X_digits = digits_.data
    y_digits = digits_.target
    X_moons = StandardScaler().fit_transform(X_moons)
    X_digits = StandardScaler().fit_transform(X_digits)

    # create train and test samples
    X_moons_train, X_moons_test, y_moons_train, y_moons_test = train_test_split(X_moons, y_moons, train_size=0.9)
    X_digits_train, X_digits_test, y_digits_train, y_digits_test = train_test_split(X_digits, y_digits, train_size=0.8)

    # create models
    moons_simple, moons_multinomial, moons_simple_reg, moons_multinomial_reg = \
        create_models(X_moons_train, y_moons_train)
    digits_simple, digits_multinomial, digits_simple_reg, digits_multinomial_reg = \
        create_models(X_digits_train, y_digits_train)

    models = [moons_simple, moons_simple_reg, moons_multinomial, moons_multinomial_reg, digits_simple,
              digits_simple_reg, digits_multinomial, digits_multinomial_reg]

    # show simple model without regularization for first dataset
    plot_decision_regions(X_moons, y_moons, clf=moons_simple)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('first dataset')
    plt.show()

    # make predicts
    print('------------------METRICS FOR MOONS------------------')
    for model in models[:4]:

        print('\n* Metrics for ', model, ' on train data *\n')
        estimate(model, X_moons_train, y_moons_train, 2)
        print('\n* Metrics for ', model, ' on test data *\n')
        estimate(model, X_moons_test, y_moons_test, 2)

    print('------------------METRICS FOR DIGITS------------------')
    for model in models[4:]:
        print('\n* Metrics for ', model, ' on train data *\n')
        estimate(model, X_digits_train, y_digits_train, 10)
        print('\n* Metrics for ', model, ' on test data *\n')
        estimate(model, X_digits_test, y_digits_test, 10)

    # grid search
    C = np.logspace(0, 4, 10)
    grid_values = [{'penalty': ['l1', 'l2'], 'C': C,
                    'solver': ['liblinear', 'saga']},
                   {'penalty': ['l2'], 'C': C,
                    'solver':['newton-cg', 'lbfgs', 'sag']}]

    grid_search_moons = GridSearchCV(LogisticRegression(max_iter=1000), grid_values, cv=5, n_jobs=-1)
    grid_search_moons.fit(X_moons_train, y_moons_train)
    print('\nOptimal model for first dataset: ', grid_search_moons.best_estimator_)
    print('Optimal parameters for first dataset: ', grid_search_moons.best_params_)
    estimate(grid_search_moons.best_estimator_, X_moons_test, y_moons_test, 2)

    grid_search_digits = GridSearchCV(LogisticRegression(max_iter=20000), grid_values, cv=5, n_jobs=-1)
    grid_search_digits.fit(X_digits_train[:200], y_digits_train[:200])
    print('\nOptimal model for second dataset: ', grid_search_digits.best_estimator_)
    print('Optimal model for second dataset: ', grid_search_digits.best_params_)
    estimate(grid_search_digits.best_estimator_, X_digits_test, y_digits_test, 10)

    # estimate models depending on train data
    estimate_size_quality(grid_search_moons.best_estimator_, X_moons, y_moons)
    estimate_size_quality(grid_search_digits.best_estimator_, X_digits, y_digits)
