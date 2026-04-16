from __future__ import annotations

from importlib import import_module
from importlib.resources import as_file, files
from pathlib import Path


def resource_filename(package_or_requirement: str, resource_name: str) -> str:
    package = import_module(package_or_requirement)
    resource = files(package).joinpath(resource_name)

    try:
        return str(Path(resource))
    except TypeError:
        with as_file(resource) as extracted:
            return str(extracted)
