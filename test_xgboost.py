from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np

def perf_parse(cm_perf):
    acc = (cm_perf[1, 1] + cm_perf[0, 0]) / np.sum(cm_perf) * 100
    recall = cm_perf[1, 1] / (cm_perf[1, 0] + cm_perf[1, 1]) * 100
    precision = cm_perf[1, 1] / (cm_perf[0, 1] + cm_perf[1, 1]) * 100
    score = 2 / ((1 / recall) + (1 / precision))

    result = [acc, recall, precision, score]
    print("Acc: {0:.3f}, Precision: {1:.3f} Recall: {2:.3f}".format(acc,  precision, recall))
    return result

if __name__ == '__main__':
    seed = 1
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('xgb.model')

    fig, ax = plt.subplots(figsize=(14, 10))
    fig_importance = xgb.plot_importance(xgb_model, importance_type='weight', title=None, ax=ax, max_num_features=15)
    fig_importance.set_xlabel("Features", fontsize=20)
    fig_importance.set_ylabel("Number of occurrences", fontsize=20)
    plt.savefig("xgboost_importance.png")
    #plt.show()

    fig, ax = plt.subplots(figsize=(28, 20))
    xgb.plot_tree(xgb_model, num_trees=0, ax=ax)
    plt.savefig("xgboost_firsttree.png")
    #plt.show()

    dataset = np.load('satellite_state.npy')
    print(dataset.shape)
    x_train, x_test, y_train, y_test = train_test_split(
        dataset[:, :-1], dataset[:, -1], random_state=seed)
    print("Train: {0:d} Test: {1:d}".format(y_train.shape[0], y_test.shape[0]))

    scaler = StandardScaler().fit(x_train)
    #x_train = scaler.transform(x_train)
    pca = PCA(n_components=2, whiten=True)
    pca.fit(x_train)

    # Train result
    predictions = xgb_model.predict(x_train)
    predictions = np.array(predictions > 0.5)
    cm_perf = (confusion_matrix(y_train, predictions))
    print(cm_perf)
    perf = perf_parse(cm_perf)

    # Test result
    predictions = xgb_model.predict(x_test)
    predictions = np.array(predictions > 0.5)
    cm_perf = (confusion_matrix(y_test, predictions))
    print(cm_perf)
    perf = perf_parse(cm_perf)

    wrong_sample = x_test[y_test != predictions]
    wrong_pred = xgb_model.predict(wrong_sample)
    print(wrong_pred)
    wrong_sample_pca = pca.transform(wrong_sample)
    plt.figure()
    colors = ['b', 'r', 'orange']
    Label_Com = ['positive', 'negative']
    for index in range(2):
        x_1 = wrong_sample_pca[y_test[y_test != predictions] == index][:, 0]
        x_2 = wrong_sample_pca[y_test[y_test != predictions] == index][:, 1]

        plt.scatter(x_1, x_2, c=colors[index], cmap='brg', alpha=0.2, marker='o', linewidth=0)
    plt.xlabel("First Principal Component", fontsize=20)
    plt.ylabel("Second Principal Component", fontsize=20)

    #plt.savefig("PCA.png")
    #plt.show()




