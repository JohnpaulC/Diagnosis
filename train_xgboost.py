from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from test_xgboost import perf_parse


def xgb_trainer(seed=1):
    dataset = np.load('satellite_state.npy')

    x_train, x_test, y_train, y_test = train_test_split(
        dataset[:, :-1], dataset[:, -1], random_state=seed)

    if False:
        # Standard preprocess the training data
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    xgb_model = xgb.XGBClassifier()
    print("Training the XGBoost classifier")
    eval_set = [(x_train, y_train), (x_test, y_test)]
    xgb_model.fit(x_train, y_train, eval_metric=["error", "rmse"], eval_set=eval_set, verbose=False)

    # retrieve performance metrics
    results = xgb_model.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)

    # plot log loss
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
    ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
    ax.legend(fontsize=15)

    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Mean Square Error', fontsize=20)
    #plt.title('XGBoost Log Loss')
    plt.savefig("MSE.png")
    plt.show()

    # plot classification error
    # fig, ax = plt.subplots(figsize=(12, 7))
    # ax.plot(x_axis, results['validation_0']['error'], label='Train')
    # ax.plot(x_axis, results['validation_1']['error'], label='Test')
    # ax.legend()
    #
    # plt.ylabel('Classification Error')
    # # plt.title('XGBoost Classification Error')
    # plt.show()

    #xgb.plot_importance(xgb_model)
    #plt.show()

    # Train result
    predictions = xgb_model.predict(x_train)
    cm_perf = (confusion_matrix(y_train, predictions))
    print(cm_perf)
    perf = perf_parse(cm_perf)

    # Test result
    predictions = xgb_model.predict(x_test)
    cm_perf = (confusion_matrix(y_test, predictions))
    print(cm_perf)
    perf = perf_parse(cm_perf)

    return xgb_model


xgb_model = xgb_trainer()
xgb_model.save_model('xgb_classifier.model')
