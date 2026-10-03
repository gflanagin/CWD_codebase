"""Compatibility wrapper around the shared land-cover parameter mapping."""

from pathlib import Path
import sys

WORKFLOW_ROOT = Path(__file__).resolve().parent
UTILITIES_DIR = WORKFLOW_ROOT / "utilities"
if str(UTILITIES_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITIES_DIR))

from shared_parameters import (  # noqa: E402
    DEFAULT_PARAMETERS,
    land_cover_to_carrying_capacity as _shared_capacity_mapping,
    load_parameters,
)


def land_cover_to_carrying_capacity(
    land_cover_array,
    spatial_parameters=None,
    parameter_file=DEFAULT_PARAMETERS,
):
    """Map class codes using the shared model parameter configuration."""
    if spatial_parameters is None:
        spatial_parameters = load_parameters(parameter_file)
    return _shared_capacity_mapping(land_cover_array, spatial_parameters)
