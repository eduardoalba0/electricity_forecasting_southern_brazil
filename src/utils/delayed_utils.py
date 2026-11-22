import gc

import numpy as np
import tensorflow as tf

def eval_single_fold(model_wrapper, model_arr, fixed_params, x_train, y_train, x_test, y_test):
    model = model_wrapper.from_array(model_arr, **fixed_params)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_true = np.ravel(y_test)
    y_p = np.ravel(y_pred)

    # Limpeza explícita da sessão Keras e liberação de memória host
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    return y_p, y_true