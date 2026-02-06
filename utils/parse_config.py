import tomllib


def get_config(path: str = "config.toml"):
    with open(path, mode="rb") as file:
        config = tomllib.load(file)

    return config
