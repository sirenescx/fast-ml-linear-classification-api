import os


def get_directory(filepath: str) -> str:
    path, _ = os.path.split(filepath)
    return path


def get_filename(filepath: str) -> str:
    _, filename = os.path.split(filepath)
    return filename


def get_filepath(directory: str, filename: str):
    return os.path.join(directory, filename)
