def _build_cache_key(model_params, features):
    params_key = tuple(sorted(model_params.items()))
    return (params_key, tuple(features))

