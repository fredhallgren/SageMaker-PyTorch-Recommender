

"""
Create config object that reads the pipeline config file

"""

from pathlib import Path
from configparser import ConfigParser

config_file = Path(__file__).parent.joinpath('pipeline.cfg')
config = ConfigParser()
config.read(config_file)

