import yaml
from pathlib import Path
import string
import argparse


def parse_args():
    """
    Parse command line arguments and override config values

    Returns
    -------
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Override config values with command line arguments"
    )

    # global
    parser.add_argument(
        "--exp_factor",
        type=int,
        default=None,
        help="expansion factor of activation channels",
    )
    parser.add_argument(
        "--input_shape",
        type=int,
        default=None,
        help="number of base activation channels",
    )

    # viz
    parser.add_argument(
        "--class_name", type=str, default=None, help="imagenet class for validation"
    )
    parser.add_argument(
        "--sample_size", type=int, default=None, help="number of samples for validation"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="number of samples for batch (applies to global and viz)",
    )
    parser.add_argument("--set", type=str, default=None, help="train or val set?")

    # sae_params
    parser.add_argument(
        "--encoder_module",
        type=str,
        default=None,
        help="what SAE encoder to use from EncoderFactory Aliases",
    )
    parser.add_argument("--topk", type=int, default=None, help="K for TopK SAE")

    return parser.parse_args()


def update_config_with_args(config, args):
    """
    Update configuration with command line arguments if provided

    Args:
        config: Dictionary containing configuration
        args: Parsed command line arguments
    """
    # Convert args to dictionary, excluding None values
    arg_dict = {k: v for k, v in vars(args).items() if v is not None}

    # Update global config if argument exists
    for key, value in arg_dict.items():
        if key in config["global"] or key in config["viz"]:
            print(f"Overriding config['global']['{key}'] with {value}")
            print(f"Overriding config['viz']['{key}'] with {value}")
            config["global"][key] = value
            config["viz"][key] = value

        elif key in config["global"]:
            print(f"Overriding config['global']['{key}'] with {value}")
            config["global"][key] = value

        elif key in config["viz"]:
            print(f"Overriding config['viz']['{key}'] with {value}")
            config["viz"][key] = value

        elif key in config["sae_params"]:
            print(f"Overriding config['sae_params']['{key}'] with {value}")
            config["sae_params"][key] = value

        elif "__" in key:
            model_type, param = key.split("__")
            if (
                model_type in config["model_zoo"]
                and param in config["model_zoo"][model_type]
            ):
                print(
                    f"Overriding config['model_zoo']['{model_type}']['{param}'] with {value}"
                )
                config["model_zoo"][model_type][param] = value

    # Update other configs if present (viz and sae params)


def load_model_zoo(config_path, globals_dict):
    """
    Load model configurations from yaml file and create model_zoo dictionary

    Args:
        config_path: Path to yaml config file
        globals_dict: Dictionary of global variables (usually pass globals())
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "nb_components" not in config["global"]:
        input_shape = config["global"]["input_shape"]
        exp_factor = config["global"]["exp_factor"]
        config["global"]["nb_components"] = int(exp_factor * input_shape)

    # Format run_name using other global parameters if it exists
    if "run_name" in config["global"]:
        template = string.Template(config["global"]["run_name"])
        # Create a copy of global config with string values for all variables
        template_vars = {k: str(v) for k, v in config["global"].items()}
        template_vars.update({k: str(v) for k, v in config["sae_params"].items()})
        for i, k in enumerate(config["model_zoo"].keys()):
            template_vars.update({f"model_{str(i+1)}": k})
        if "target_class" not in template_vars:
            template_vars.update({"target_class": "ALL"})
            config["global"]["target_class"] = "ALL"
        else:
            template_vars.update(
                {"target_class": config["global"]["target_class"].replace(" ", "")}
            )
        config["global"]["run_name"] = template.substitute(template_vars)

    # Replace model_type strings with actual classes
    for model_config in config["model_zoo"].values():
        model_class = globals_dict.get(model_config["og_model"])
        if model_class is None:
            raise ValueError(
                f"Model class {model_config['og_model']} not found in global scope"
            )

        # assign actual class to model_zoo og_model key
        model_config["og_model"] = model_class

    return config["global"], config["viz"], config["sae_params"], config["model_zoo"]
