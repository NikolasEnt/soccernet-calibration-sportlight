from argus import Model


def load_compatible_weights(old_model: Model, new_model: Model) -> Model:
    """Load only compatible weights from older model.

    It can be useful if the new model is slighly different (e.g. changed number
    of classes).

    Args:
        old_model (Model): Original argus model, which is used as the base.
        new_model (Model): The new argus models to take new weights from.

    Returns:
        Model: Resulted argus model.
    """
    state_dict_old = old_model.nn_module.state_dict()
    state_dict_new = new_model.nn_module.state_dict()
    for k in state_dict_old:
        if k in state_dict_new\
                and (state_dict_old[k].shape != state_dict_new[k].shape):
            print(f'Skip loading parameter {k}, '
                  f'required shape {state_dict_new[k].shape}, '
                  f'loaded shape {state_dict_old[k].shape}.')
            state_dict_old[k] = state_dict_new[k]

    for k in state_dict_new:
        if k not in state_dict_old:
            print(f'No param in old: {k}.')
            state_dict_old[k] = state_dict_new[k]

    new_model.nn_module.load_state_dict(state_dict_old)
    return new_model
