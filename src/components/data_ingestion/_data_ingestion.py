import os
from urllib import request

from pathlib import Path
from config import ConfigManager, ParamManager

from dataclasses import dataclass
from ensure import ensure_annotations

from reytools.logger import logging
from reytools.file_system import get_file_size
from reytools.file_system import create_directories
from reytools.file_system import unzip_file


@dataclass(frozen=True)
class DataIngestionConfig:
    movielens_url: str
    movielens_local_dir: Path
    movielens_zip_file: Path
    lfm_url: str
    lfm_local_dir: Path
    lfm_zip_file: Path
    data_root_dir: Path
    artifact_root: Path
    movielens_name: str
    lfm_data_name: str


class DataIngestionConfManager(ConfigManager):
    def __init__(self):
        cur_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        super().__init__(cur_dir)

    def get_configuration(self) -> DataIngestionConfig:
        config = self.config

        box_config = DataIngestionConfig(
            movielens_url=config.movielens_url,
            movielens_local_dir=Path(config.movielens_local_dir),
            movielens_zip_file=Path(config.movielens_zip_file),
            lfm_url=config.lfm_url,
            lfm_local_dir=Path(config.lfm_local_dir),
            lfm_zip_file=Path(config.lfm_zip_file),
            data_root_dir=Path(config.data_root_dir),
            artifact_root=Path(config.artifact_root),
            movielens_name=config.movielens_name,
            lfm_data_name=config.lfm_data_name,
        )

        logging.info("DataIngestion configuration loaded")
        return box_config


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfManager().get_configuration()
        self._create_directories()

    def _create_directories(self):
        create_directories([self.config.movielens_local_dir, self.config.lfm_local_dir])

    def download_extract_movielens(self):
        self._download_movie_lense()
        self._extract_movie_lens()

    def _download_movie_lense(self):
        url = self.config.movielens_url
        file_name = self.config.movielens_zip_file
        if not os.path.exists(file_name):
            file_name, header = request.urlretrieve(url=url, filename=file_name)
            logging.info(f"{file_name} downloaded with following info {header}")
        else:
            size_of_file = get_file_size(file_name)
            logging.info(f"file already exists in: {file_name} with size: {size_of_file}")

    def _extract_movie_lens(self):
        """Extact the movie lense zip file of the dataset into the data directory"""
        unzip_file_dir = self.config.movielens_local_dir
        zip_data_dir = self.config.movielens_zip_file
        create_directories([unzip_file_dir])
        unzip_file(zip_data_dir, unzip_file_dir)
