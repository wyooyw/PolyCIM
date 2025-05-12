"""
{
  "chip_config": {
    "core_config": {
        "cim_unit_config": {
            "macro_total_cnt": 8,
            "macro_group_size": 1,
            "macro_size": {
            "compartment_cnt_per_macro": 32,
            "element_cnt_per_compartment": 8,
            "row_cnt_per_element": 1,
            "_comment": "bit-width-per-weight",
            "bit_width_per_row": 8
            },
    }
    "address_space_config": [
      {"name": "cim_unit", "size": 2048},
    ]
  }
}
"""

import json
import os

cim_unit_configs = {
    "g8m8c64b64": {
        "cim_unit_config": {
            "macro_total_cnt": 8,
            "compartment_cnt_per_macro": 64,
            "element_cnt_per_compartment": 8,
            "bit_width_per_row": 8
        },
        "cim_unit_size": 4096
    },
    "g8m8c32b64": {
        "cim_unit_config": {
            "macro_total_cnt": 8,
            "compartment_cnt_per_macro": 32,
            "element_cnt_per_compartment": 8,
            "bit_width_per_row": 8,
        },
        "cim_unit_size": 2048
    },
    "g8m8c16b32": {
        "cim_unit_config": {
            "macro_total_cnt": 8,
            "compartment_cnt_per_macro": 16,
            "element_cnt_per_compartment": 4,
            "bit_width_per_row": 8
        },
        "cim_unit_size": 512
    }
}

template_config_path = "/app/CIMCompiler/PolyCIM/polycim/exp/iccad25/cimsim_configs/template.json"
output_dir = "/app/CIMCompiler/PolyCIM/polycim/exp/iccad25/cimsim_configs"

def generate_configs():
    # Read the template config
    with open(template_config_path, 'r') as f:
        template_config = json.load(f)
    
    # Generate a config file for each configuration
    for config_name, config_values in cim_unit_configs.items():
        # Make a deep copy of the template
        updated_config = json.loads(json.dumps(template_config))
        
        # Update cim_unit_config
        cim_config = config_values["cim_unit_config"]
        updated_config["chip_config"]["core_config"]["cim_unit_config"]["macro_total_cnt"] = cim_config["macro_total_cnt"]
        updated_config["chip_config"]["core_config"]["cim_unit_config"]["macro_size"]["compartment_cnt_per_macro"] = cim_config["compartment_cnt_per_macro"]
        updated_config["chip_config"]["core_config"]["cim_unit_config"]["macro_size"]["element_cnt_per_compartment"] = cim_config["element_cnt_per_compartment"]
        updated_config["chip_config"]["core_config"]["cim_unit_config"]["macro_size"]["bit_width_per_row"] = cim_config["bit_width_per_row"]
        
        # Update address_space_config
        for address_space in updated_config["chip_config"]["address_space_config"]:
            if address_space["name"] == "cim_unit":
                address_space["size"] = config_values["cim_unit_size"]
        
        # Write the updated config to a file
        output_path = os.path.join(output_dir, f"{config_name}.json")
        with open(output_path, 'w') as f:
            json.dump(updated_config, f, indent=2)
        
        print(f"Generated config file: {output_path}")

if __name__ == "__main__":
    generate_configs()


