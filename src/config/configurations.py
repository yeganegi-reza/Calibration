import os
from abc import abstractmethod
from pathlib import Path
from ensure import ensure_annotations

from reytools.read_write import read_yaml
from reytools.logger import logging


class ConfigManager:
    @ensure_annotations
    def __init__(self, comp_dir: Path):
        self.__set_config_file_path(comp_dir)
        self.__read_configuration()

    @ensure_annotations
    def __read_configuration(self):
        self.config = read_yaml(self.config_file_path)

    @ensure_annotations
    def __set_config_file_path(self, cur_dir: Path):
        config_file_path = Path(os.path.join(cur_dir, "config", "config.yaml"))
        self.config_file_path = config_file_path

    @abstractmethod
    def get_configuration(self):
        raise NotImplemented


class ParamManager:
    @ensure_annotations
    def __init__(self, comp_dir: Path):
        self.__set_param_file_path(comp_dir)
        self.__read_params()

    @ensure_annotations
    def __read_params(self):
        self.params = read_yaml(self.param_file_path)

    @ensure_annotations
    def __set_param_file_path(self, cur_dir: Path):
        param_file_path = Path(os.path.join(cur_dir, "params.yaml"))
        self.param_file_path = param_file_path

    @abstractmethod
    def get_params(self):
        raise NotImplemented
