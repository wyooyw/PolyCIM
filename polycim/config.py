from dataclasses import dataclass
import json
import os
import copy

@dataclass
class CIMConfig:
    n_row: int
    n_group_vcol: int
    n_comp: int
    n_group: int
    n_macro_per_group: int
    n_macro: int

_raw_config = None
def set_raw_config(config):
    global _raw_config
    if _raw_config is not None:
        raise ValueError("config already set")
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")
    _raw_config = copy.deepcopy(config)

def set_raw_config_by_path(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    set_raw_config(config)

def get_raw_config():
    global _raw_config
    if _raw_config is None:
        raise ValueError("config not set")
    return _raw_config

# def get_raw_config():
#     global config
#     if config is None:
#         CONFIG_PATH = os.environ.get("CONFIG_PATH")
#         config_json_path = os.path.join(CONFIG_PATH)
#         with open(config_json_path, "r") as f:
#             config = json.load(f)
#     return config

def get_config():
    config = get_raw_config()

    config_macro = config["macro"]
    n_group = config_macro["n_group"]
    n_row = config_macro["n_row"]
    n_macro = config_macro["n_macro"]
    assert n_macro >= n_group and n_macro % n_group == 0, f"{n_macro=}, {n_group=}"

    n_macro_per_group = n_macro // n_group

    n_bcol = config_macro["n_bcol"]
    n_vcol = n_bcol // 8
    n_group_vcol = n_vcol * n_macro_per_group

    n_comp = config_macro["n_comp"]
    n_macro_per_group=n_macro // n_group
    return CIMConfig(
        n_row=n_row,
        n_group_vcol=n_group_vcol,
        n_comp=n_comp,
        n_group=n_group,
        n_macro_per_group=n_macro_per_group,
        n_macro=n_macro
    )

def get_memory_sizes():
    """
    "memory_list": [
        {
            "name": "macro",
            "type": "macro",
            "addressing": {
                "offset_byte": 0,
                "size_byte": 1048576,
                "comments": "offset: 0 Byte, size: 128 * 1024 Byte"
            }
        },
        {
    """
    config = get_raw_config()

    memory_list = config["memory_list"]
    memory_type_to_sizes = {}
    for memory in memory_list:
        big_name = "__"+memory["name"].upper()+"__"
        memory_type_to_sizes[big_name] = memory["addressing"]["size_byte"]
    return memory_type_to_sizes

def get_memory_base(memory_type):
    config = get_raw_config()

    memory_list = config["memory_list"]
    for memory in memory_list:
        if memory["name"]==memory_type:
            return memory["addressing"]["offset_byte"]
    assert False, f"{memory_type=} not found"

def get_memory_types():
    config = get_raw_config()
    memory_list = config["memory_list"]
    memory_types = []
    for memory in memory_list:
        big_name = "__"+memory["name"].upper()+"__"
        memory_types.append(big_name)
    return memory_types