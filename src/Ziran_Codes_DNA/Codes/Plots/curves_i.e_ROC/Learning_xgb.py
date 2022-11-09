from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import mean_squared_error
from matplotlib.legend_handler import HandlerLine2D
def ROC_curve(model, x_test, y_test):
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax, alpha=0.8)
    plt.show()

def learning_curve(list_nb_trees, x_train, y_train, x_test, y_test):
    train_results = []
    test_results = []
    for nb_trees in list_nb_trees:
        xg = XGBClassifier(n_estimators=nb_trees)
        xg.fit(x_train, y_train)

        train_results.append(mean_squared_error(y_train, xg.predict(x_train)))
        test_results.append(mean_squared_error(y_test, xg.predict(x_test)))

    line1, = plt.plot(list_nb_trees, train_results, color="r", label="Training Score")
    line2, = plt.plot(list_nb_trees, test_results, color="g", label="Testing Score")

    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('MSE')
    plt.xlabel('n_estimators')
    plt.show()
