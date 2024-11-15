from dataclasses import dataclass
import json

@dataclass
class CIMConfig:
    n_row: int
    n_group_vcol: int
    n_comp: int
    n_group: int

config_json_path = "/home/wangyiou/Desktop/pim_compiler/playground/config.json"
with open(config_json_path, "r") as f:
    config = json.load(f)

def get_config():
    global config

    config_macro = config["macro"]
    
    n_group = 16
    n_row = config_macro["n_row"]
    n_macro = config_macro["n_macro"]
    n_macro_per_group = n_macro // n_group

    n_bcol = config_macro["n_bcol"]
    n_vcol = n_bcol // 8
    n_group_vcol = n_vcol * n_macro_per_group

    n_comp = config_macro["n_comp"]

    return CIMConfig(
        n_row=n_row,
        n_group_vcol=n_group_vcol,
        n_comp=n_comp,
        n_group=n_group
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
    global config

    memory_list = config["memory_list"]
    memory_type_to_sizes = {}
    for memory in memory_list:
        big_name = "__"+memory["name"].upper()+"__"
        memory_type_to_sizes[big_name] = memory["addressing"]["size_byte"]
    return memory_type_to_sizes

def get_memory_base(memory_type):
    global config

    memory_list = config["memory_list"]
    for memory in memory_list:
        if memory["name"]==memory_type:
            return memory["addressing"]["offset_byte"]
    assert False, f"{memory_type=} not found"


def get_memory_size(memory_type):
    global config

    memory_list = config["memory_list"]
    for memory in memory_list:
        if memory["name"]==memory_type:
            return memory["addressing"]["size_byte"]
    assert False, f"{memory_type=} not found"