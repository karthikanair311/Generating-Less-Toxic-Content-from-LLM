import logging
import os
import yaml


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def load_config_from_yaml():
    """
    Load the configuration YAML file and return the configuration dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Configuration dictionary loaded from the YAML file.
    """
    file_path = "assets/config/config.yaml"
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
        try:
            if os.environ.get("image_table_bucket"):
                logger.info(
                    "Environment variable for image_table_bucket found")
                config["Table_args"]["bucket_name"] = os.environ.get(
                    "image_table_bucket")
        except Exception as e:
            logger.error(e)

    return config


# Load the configuration file
# config = load_config()
config = load_config_from_yaml()