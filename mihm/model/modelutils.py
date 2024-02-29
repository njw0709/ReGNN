from collections import OrderedDict


def get_index_prediction_weights(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "layer" in k or "bn" in k:
            new_state_dict[k] = v

    return new_state_dict
