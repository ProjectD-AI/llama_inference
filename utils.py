import json
import sys
from argparse import Namespace
import torch
from model.llama import NormalLinear


def load_hyperparam(default_args):
    """
    Load arguments form argparse and config file
    Priority: default options < config file < command line args
    """
    with open(default_args.config_path, mode="r", encoding="utf-8") as f:
        config_args_dict = json.load(f)

    default_args_dict = vars(default_args)

    command_line_args_dict = {k: default_args_dict[k] for k in [
        a[2:] for a in sys.argv if (a[:2] == "--" and "local_rank" not in a)
    ]}
    default_args_dict.update(config_args_dict)
    default_args_dict.update(command_line_args_dict)
    args = Namespace(**default_args_dict)

    return args


def _load_state_dict_into_model(model_to_load, model_path, start_prefix=""):
    # Convert old format to new format if needed from a PyTorch state_dict

    # copy state_dict so _load_from_state_dict can modify it
    state_dict = torch.load(model_path, map_location="cpu")
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    state_dict['target.lm.weight'] = state_dict['target.lm.output_layer.weight']
    del state_dict['target.lm.output_layer.weight']
    state_dict['embedding.embedding.weight'] = state_dict['embedding.word.embedding.weight']
    del state_dict['embedding.word.embedding.weight']

    if metadata is not None:
        metadata['embedding.embedding'] = metadata['embedding.word.embedding']
        metadata['target.lm'] = metadata['target.lm.output_layer']
        if metadata.get('embedding.dropout', None) is not None:
            del metadata['embedding.dropout']
        del metadata['embedding.word']
        del metadata['embedding.word.embedding']
        del metadata['target.lm.output_layer']
        del metadata['target.lm.softmax']
        del metadata['target.lm.criterion']
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            import deepspeed
            # In sharded models, each shard has only part of the full state_dict, so only gather
            # parameters that are in the current state_dict.
            named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
            params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
            if len(params_to_gather) > 0:
                # because zero3 puts placeholders in model params, this context
                # manager gathers (unpartitions) the params of the current layer, then loads from
                # the state dict and then re-partitions them again
                with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    load(model_to_load, state_dict, prefix=start_prefix)
    # Delete `state_dict` so it could be collected by GC earlier. Note that `state_dict` is a copy of the argument, so
    # it's safe to delete it.
    del state_dict

    return model_to_load


def convert_normal_parameter_to_int8(model, threshold=6.0, modules_to_not_convert=None, current_key_name=None):
    import bitsandbytes as bnb
    modules_to_not_convert = ["lm"] if modules_to_not_convert is None else modules_to_not_convert
    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if len(list(module.children())) > 0:
            convert_normal_parameter_to_int8(module, threshold, modules_to_not_convert, current_key_name)

        if isinstance(module, bnb.nn.Linear8bitLt) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            if not any(key in ".".join(current_key_name) for key in modules_to_not_convert):
                model._modules[name].weight = bnb.nn.Int8Params(
                    module.weight.data,
                    requires_grad=False,
                    has_fp16_weights=False
                )
                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model
