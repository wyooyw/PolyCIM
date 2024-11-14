from dataclasses import dataclass
import json

@dataclass
class CIMConfig:
    n_row: int
    n_group_vcol: int
    n_comp: int
    n_group: int


def get_config():
    config_json_path = "/home/wangyiou/Desktop/pim_compiler/playground/config.json"
    with open(config_json_path, "r") as f:
        config = json.load(f)
    config = config["macro"]
    
    n_group = 16
    n_row = config["n_row"]
    n_macro = config["n_macro"]
    n_macro_per_group = n_macro // n_group

    n_bcol = config["n_bcol"]
    n_vcol = n_bcol // 8
    n_group_vcol = n_vcol * n_macro_per_group

    n_comp = config["n_comp"]

    return CIMConfig(
        n_row=n_row,
        n_group_vcol=n_group_vcol,
        n_comp=n_comp,
        n_group=n_group
    )