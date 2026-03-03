import numpy as np
from pymatgen.core import Structure

def wrap_coords(cartesian_coords: np.ndarray, lattice_matrix: np.ndarray) -> np.ndarray:
    """Wrap Cartesian coordinates into the unit cell (orthogonal), aligns the mdtraj convention.

    Args:
        cartesian_coords: Cartesian coordinates of shape (N, 3).
        lattice_matrix: 3x3 lattice matrix.

    Returns:
        np.ndarray: Wrapped coordinates scaled by lattice vector norms.
    """
    struct = Structure(lattice_matrix,
                    ["Li"] * len(cartesian_coords),
                    cartesian_coords, coords_are_cartesian=True)
    # aligns the mdtraj convention
    return struct.frac_coords * np.sqrt((lattice_matrix ** 2).sum(axis=1))

def wrap_supercell_lpscl_ii(
    scaling_mat: np.ndarray,
    super_coords: dict,
    super_centers: dict,
) -> tuple[dict, dict]:
    """Wrap supercell coordinates to the unit cell (orthogonal), aligns the mdtraj convention.
    Note: This wrapper is only necessary for the trajs saved in mdtraj h5 format.

    Args:
        scaling_mat: Scaling matrix for the supercell.
        super_coords: Supercell coordinates.
        super_centers: Supercell cluster centers.

    Returns:
        tuple[dict, dict]: Updated `super_centers` and `super_coords`.
    """
    pm = Structure.from_file(f'../data/structures/lpscl-ii.cif')

    pymat_super = pm.make_supercell(scaling_mat)
    wy = list(super_centers.keys())
    num_cluster = super_coords[wy[0]][0].shape[0]
    num_sites = super_coords[wy[0]][0].shape[1]
    lattice_matrix = pymat_super.lattice.matrix

    for center, coords in super_coords[wy[0]].items():
        super_coords['24g'][center] = wrap_coords(coords.reshape(num_cluster * num_sites, 3),
         lattice_matrix).reshape(num_cluster, num_sites, 3)

    for num in range(len(super_centers['24g'])):
        coords = super_centers['24g'][num]
        super_centers['24g'][num] = wrap_coords(coords, lattice_matrix).reshape(num_cluster, 3)

    return super_centers, super_coords