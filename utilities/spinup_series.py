"""XDMF time series for the disease spin-up.

t = 0 is the seeding and the final frame is the control run's initial state.
Written only when the spin-up runs rather than loading its cache.
"""

from __future__ import annotations

from dolfinx import io
from dolfinx.fem import Function

# Mixed-space order.
LABELS = (("S", "Susceptible"), ("E", "Exposed"), ("I", "Preclinical"),
          ("R", "Clinical"), ("W", "Environment"))


class SpinupSeriesWriter:
    """Writes the five compartments of a mixed state function each frame."""

    def __init__(self, domain, scalar_space, folder):
        folder.mkdir(parents=True, exist_ok=True)
        self.files = {}
        self.fields = {}
        for key, label in LABELS:
            handle = io.XDMFFile(
                domain.comm, str(folder / f"spinup_{key}.xdmf"), "w"
            )
            handle.write_mesh(domain)
            self.files[key] = handle
            self.fields[key] = Function(scalar_space, name=label)
        self.frames = 0

    def write(self, state_function, years):
        for index, (key, _label) in enumerate(LABELS):
            self.fields[key].interpolate(state_function.sub(index))
            self.files[key].write_function(self.fields[key], float(years))
        self.frames += 1

    def close(self):
        for handle in self.files.values():
            handle.close()
