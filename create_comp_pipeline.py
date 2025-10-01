from pathlib import Path
import os


def create_comp(comp_name, comp_file_path):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = Path(os.path.join(cur_dir, "comp.txt"))
    with open(file_path, "r") as file:
        all_file_content = file.read()

    all_file_content = all_file_content.replace("{comp_name}", comp_name)
    with open(comp_file_path, "w") as file:
        file.write(all_file_content)


def create_pipeline(comp_name, pipleline_file_path):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = Path(os.path.join(cur_dir, "pipeline.txt"))
    with open(file_path, "r") as file:
        all_file_content = file.read()

    all_file_content = all_file_content.replace("{comp_name}", comp_name)
    with open(pipleline_file_path, "w") as file:
        file.write(all_file_content)


def create_comp_dir(comp_dir):
    os.makedirs(comp_dir, exist_ok=True)


def create_conf_file(comp_dir):
    config_dir = Path(os.path.join(comp_dir, "config"))
    os.makedirs(config_dir, exist_ok=True)
    conf_file_path = Path(os.path.join(config_dir, "config.yaml"))
    with open(conf_file_path, "w") as file:
        pass


def create_and_init_file(comp_dir):
    conf_file_path = Path(os.path.join(comp_dir, "params.yaml"))
    with open(conf_file_path, "w") as file:
        pass
    conf_file_path = Path(os.path.join(comp_dir, "__init__.py"))
    with open(conf_file_path, "w") as file:
        pass


def create_pipeline_dir(pipeline_dir_path):
    os.makedirs(pipeline_dir_path, exist_ok=True)


def check_comp_name(comp_name: str):
    splits = comp_name.split(" ")
    cap_splits = [s.capitalize() for s in splits]
    file_name = splits[0]
    for i in range(1, len(splits)):
        file_name += "_" + splits[i]
    class_name = "".join(cap_splits)
    return file_name, class_name


cur_dir = os.path.dirname(os.path.abspath(__file__))
comp_name = "data ingestion"

file_name, class_name = check_comp_name(comp_name=comp_name)
comp_dir = Path(os.path.join(cur_dir, "src", "components", file_name))
comp_file_path = Path(os.path.join(comp_dir, f"_{file_name}.py"))
pipeline_dir_path = Path(os.path.join(cur_dir, "src", "pipeline"))
pipeline_file_path = Path(os.path.join(pipeline_dir_path, f"stage_{file_name}.py"))
create_pipeline_dir(pipeline_dir_path)
create_comp_dir(comp_dir)
create_conf_file(comp_dir)
create_and_init_file(comp_dir)
create_comp(class_name, comp_file_path)
create_pipeline(class_name, pipeline_file_path)
