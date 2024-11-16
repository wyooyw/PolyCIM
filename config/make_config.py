import json
import os


def make_config(n_macro_per_group, name):
    """
    1.read config_template.json
    2.calculate each memory's offset, fill into 'offset_byte'
    3.write config.json
    """
    import json

    config_template_path = os.path.join(
        os.path.dirname(__file__), "config_template.json"
    )
    config_path = os.path.join(os.path.dirname(__file__), f"{name}.json")
    with open(config_template_path, "r") as f:
        config = json.load(f)

    # config macro
    n_macro = n_macro_per_group * config["macro"]["n_group"]
    config["macro"]["n_macro"] = n_macro
    macro_size = (
        config["macro"]["n_macro"] 
        * config["macro"]["n_row"] 
        * config["macro"]["n_comp"] 
        * (config["macro"]["n_bcol"] // 8)
    ) 
    assert config["memory_list"][0]["name"]=="macro"
    config["memory_list"][0]["addressing"]["size_byte"] = macro_size

    # config memory
    last_end = 0
    for memory in config["memory_list"]:
        addressing = memory["addressing"]
        addressing["offset_byte"] = last_end
        last_end += addressing["size_byte"]
    

    if os.path.exists(config_path):
        make_sure = input(
            f"{name}.json already exists, do you want to overwrite it? (y/n)"
        )
        if make_sure.lower() == "n":
            return
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"make {name}.json done")


if __name__ == "__main__":
    for n_macro_per_group in [4,8,12,16]:
        make_config(n_macro_per_group, f"config_gs_{n_macro_per_group}")
