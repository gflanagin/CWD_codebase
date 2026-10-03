"""Verify imports and versions needed by the mesh workflow and CWD solver."""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
import sys


IMPORTS = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("rasterio", "rasterio"),
    ("meshio", "meshio"),
    ("gmsh", "gmsh"),
    ("pyvista", "pyvista"),
    ("mmgpy", "mmgpy"),
    ("mpi4py", "mpi4py"),
    ("petsc4py", "petsc4py"),
    ("basix", "fenics-basix"),
    ("ufl", "fenics-ufl"),
    ("dolfinx", "fenics-dolfinx"),
)


def package_version(distribution):
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def main():
    failures = []
    print(f"Python {sys.version.split()[0]}")
    for module_name, distribution in IMPORTS:
        try:
            importlib.import_module(module_name)
            print(f"[OK] {module_name:<10} {package_version(distribution)}")
        except Exception as error:
            failures.append((module_name, error))
            print(f"[FAIL] {module_name:<10} {error}")

    if not failures:
        import pyvista as pv
        import mmgpy  # noqa: F401 - registers the PyVista MMG accessor

        if not hasattr(pv.PolyData(), "mmg"):
            failures.append(("mmgpy accessor", "PyVista .mmg accessor is unavailable"))
            print("[FAIL] PyVista .mmg accessor is unavailable")
        else:
            print("[OK] PyVista .mmg accessor registered")

    if failures:
        print(f"\nEnvironment verification failed ({len(failures)} problem(s)).")
        return 1

    print("\nEnvironment is ready for the mesh workflow and solver.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
