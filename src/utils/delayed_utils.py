import numpy as np

def eval_single_fold(model_wrapper, model_arr, fixed_params, x_train, y_train, x_test, y_test):
    model = model_wrapper.from_array(model_arr, **fixed_params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return np.ravel(y_pred), np.ravel(y_test)