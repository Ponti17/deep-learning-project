# Core
import yaml

def update_accumulated_output(accumulated_output, step_output):
    """
    Just makes our life easier when we want to accumulate the output of the
    validation.
    """
    step_output = step_output["raw"]

    for key, step_value in step_output.items():
        if key in accumulated_output:
            accumulated_output[key].extend(list(step_value))
        else:
            accumulated_output[key] = list(step_value)
    return


def read_yaml(yaml_path):
    """
    Read a YAML file :)
    """
    with open(yaml_path, "r", encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    return config


def dump_yaml(save_path, dictionary):
    """
    Dump it :(
    """
    with open(save_path, "w", encoding='utf-8') as f:
        yaml.dump(
            dictionary,
            f,
            default_flow_style=False,
            sort_keys=False
        )
