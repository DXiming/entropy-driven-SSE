"""
Clustering lithium atoms in SSEs using K-means.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from kneed import KneeLocator
from sklearn.cluster import KMeans

from pymatgen.core import Structure
from pyxtal import pyxtal
from pyxtal.symmetry import Group




def _find_optimal_k(
    atomic_positions: np.ndarray,
    kmeans_cls: type = KMeans,
    **kmeans_kwargs: Any,
) -> int:
    """Find the optimal number of clusters using the elbow (knee) method.

    Args:
        atomic_positions : Coordinate array of shape (N, 3).
        kmeans_cls : KMeans-compatible class to use for clustering.
        **kmeans_kwargs : Extra keyword arguments forwarded to *kmeans_cls*.
    
    Returns:
        n_points: Optimal number of clusters.
    """
    n_points = len(atomic_positions)
    if n_points <= 1:
        return n_points

    wcss = []
    for k in range(1, n_points):
        model = kmeans_cls(n_clusters=k, **kmeans_kwargs).fit(atomic_positions)
        wcss.append(model.inertia_)

    knee_locator = KneeLocator(
        range(1, n_points),
        wcss,
        S=1.0,
        curve='convex',
        direction='decreasing',
    )
    if knee_locator.knee is not None:
        return knee_locator.knee
    return n_points


class ClusterLi():
    """Classification based on Wyckoff sites and unsupervised K-means clustering."""

    def __init__(self, unit_struc_file: str, save_dir: str) -> None:
        super(ClusterLi, self).__init__()
        self.save_dir = save_dir
        self.file_name = os.path.basename(unit_struc_file)
        self.pmg = Structure.from_file(unit_struc_file)
        self.pyx = pyxtal()

    def WyckoffSites(self) -> tuple[int, list[str]]:
        """Return lithium sites classified in Wyckoff expression.

        Returns:
            sp_group: Space group number
            wp_li: List of unique Wyckoff labels for Li
        """
        self.pyx.from_seed(self.pmg)
        sp_group = self.pyx.group.number
        wp_li = self.pyx.get_site_labels()['Li']
        return sp_group, list(set(wp_li))

    def WyClassCoords(self, merge: bool = False, save: bool = True) -> dict[str, np.ndarray]:
        """Return lithium coordinates fitted to all possible Wyckoff sites.

        Args:
            merge: If True, merge all Wyckoff sites into the first label.
            save: If True, save the unit cell structure to *save_dir*.

        Returns:
            sites_list: Mapping from Wyckoff label to fractional coordinates.
        """
        pmg_li = self.pmg.copy()
        if save:
            pmg_li.to(f"{self.save_dir}/{self.file_name}", fmt="poscar")
        species_rm = [str(i) for i in pmg_li.elements]
        species_rm.remove('Li')
        pmg_li = pmg_li.remove_species(species=species_rm)
        coords = pmg_li.frac_coords
        sp_group, wp_li = self.WyckoffSites()
        g = Group(sp_group, dim=3, use_hall=False)

        print(f"{len(coords)} Li atoms in total.")
        sites_list: dict[str, np.ndarray] = {}
        if merge:
            sites_list[wp_li[0]] = np.array(coords)
            return sites_list
        else:
            if len(wp_li) == 1:
                sites_list[wp_li[0]] = np.array(coords)
                return sites_list
            else:
                for wp in wp_li:
                    sites_list[wp] = []
                    check_wp = 0
                    for num in range(len(coords)):
                        sp = g[wp].get_all_positions(coords[num])
                        if sp is not None:
                            sites_list[wp].append(coords[num])
                        else:
                            check_wp += 1
                if len(sites_list[wp]) == len(coords):
                    print(f"other wp sites in this \'{wp}\', merged together")
                    updated_list = {}
                    updated_list[wp] = sites_list[wp]
                    return updated_list
                elif check_wp == len(coords):
                    del sites_list
                    sites_list = {}
                    sites_list[wp_li[0]] = np.array(coords)
                    return sites_list
        return sites_list

    def KClassCenters(self, plot: bool = False) -> dict[str, np.ndarray]:
        """Return further classified lithium cluster centers through K-means."""
        
        sites_list = self.WyClassCoords(save=self.save_dir)
        clusters: dict[str, np.ndarray] = {}
        for wp, coords in sites_list.items():
            atomic_positions = np.array(coords)

            optimal_k = _find_optimal_k(atomic_positions, random_state=0)
            kmeans = KMeans(n_clusters=optimal_k, random_state=0).fit(atomic_positions)

            if plot:
                labels = kmeans.labels_
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for i in range(optimal_k):
                    ax.scatter(*atomic_positions[labels == i].T, label=f'Cluster {i+1}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.legend()
                plt.show()

            clusters[wp] = np.array(kmeans.cluster_centers_)
        return clusters

    def transform_to_supercell(
        self,
        unit_cell_coords: np.ndarray,
        lattice_vectors: np.ndarray,
        A: int,
        B: int,
        C: int,
    ) -> np.ndarray:
        """Transform unit cell sites to supercell with size A x B x C.

        Args:
            unit_cell_coords: Cartesian coordinates of shape (N, 3).
            lattice_vectors: 3x3 lattice matrix.
            A, B, C: Supercell dimensions.

        Returns:
            supercell_coords: Supercell coordinates.
        """
        N = unit_cell_coords.shape[0]

        i_indices = np.arange(A)
        j_indices = np.arange(B)
        k_indices = np.arange(C)

        i_grid, j_grid, k_grid = np.meshgrid(i_indices, j_indices, k_indices, indexing='ij')

        unit_cell_coords_reshaped = unit_cell_coords[:, None, None, None, :]

        i_grid_expanded = i_grid[:, :, :, np.newaxis]
        j_grid_expanded = j_grid[:, :, :, np.newaxis]
        k_grid_expanded = k_grid[:, :, :, np.newaxis]

        displacement_vectors = (i_grid_expanded * lattice_vectors[0, :]) + \
                             (j_grid_expanded * lattice_vectors[1, :]) + \
                             (k_grid_expanded * lattice_vectors[2, :])

        supercell_coords = unit_cell_coords_reshaped + displacement_vectors
        supercell_coords = supercell_coords.reshape(N * A * B * C, 3)

        return supercell_coords

    def KClassCenterSuper(
        self, A: int, B: int, C: int, plot: bool = False, frac: bool = False,
    ) -> dict[str, list[np.ndarray]]:
        """Return cluster centers in supercell with size A x B x C.

        Args:
            A, B, C: Supercell dimensions.
            plot: If True, show a 3-D scatter plot.
            frac: If True, return fractional coordinates.

        Returns:
            clusters_super: Mapping from Wyckoff label to list of supercell centers.
        """
        clusters = self.KClassCenters(plot=False)
        cell = self.pmg.lattice.matrix
        clusters_super: dict[str, list[np.ndarray]] = {}
        for wp, centers in clusters.items():
            centers = np.dot(centers, cell.T)
            clusters_super[wp] = []
            for coord in centers:
                cart_centers = self.transform_to_supercell(np.array([coord]), cell, A, B, C)
                if not frac:
                    clusters_super[wp].append(cart_centers)
                else:
                    supercell_matrix = np.array([A * cell[0], B * cell[1], C * cell[2]])
                    frac_centers = cart_centers @ np.linalg.inv(supercell_matrix)
                    clusters_super[wp].append(frac_centers)

        if plot:
            for wp, centers in clusters_super.items():
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for i in range(len(clusters[wp])):
                    ax.scatter(*clusters_super[wp][i].T, label=f'Cluster {i+1}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.legend()
                plt.show()
        return clusters_super

def periodic_distance(
    coord1: np.ndarray,
    coord2: np.ndarray,
    lattice_vectors: np.ndarray,
) -> float:
    """Minimum image distance between two Cartesian points under PBC."""
    inv_lattice = np.linalg.inv(lattice_vectors)
    frac_diff = np.dot(coord1 - coord2, inv_lattice)
    frac_diff -= np.round(frac_diff)
    diff_cart = np.dot(frac_diff, lattice_vectors)
    return np.linalg.norm(diff_cart)


class PBCKmeans():
    """Cluster lithiums through k-means under PBC conditions."""

    def __init__(
        self,
        coords: dict[str, np.ndarray],
        cell: np.ndarray,
        cart: bool = False,
        plot: bool = False,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 0.0001,
        random_state: int | None = 0,
    ) -> None:
        super(PBCKmeans, self).__init__()
        self.coords = coords
        self.cell = cell
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fractional_to_cartesian(self, fractional_coords: np.ndarray, cell: np.ndarray | None = None) -> np.ndarray:
        """Convert fractional coordinates to Cartesian coordinates."""

        if cell is not None:
            return np.dot(fractional_coords, cell)
        return np.dot(fractional_coords, self.cell)

    def cartesian_to_fractional(self, cartesian_coords: np.ndarray, cell: np.ndarray | None = None) -> np.ndarray:
        """Convert Cartesian coordinates to fractional coordinates."""

        if cell is not None:
            inv_lattice_vectors = np.linalg.inv(cell)
        else:
            inv_lattice_vectors = np.linalg.inv(self.cell)
        return np.dot(cartesian_coords, inv_lattice_vectors)

    def KClassCenters(self, plot: bool = False) -> tuple[dict, dict, dict, dict, dict]:
        """K-means classification of lithium atoms."""

        def calculate_intra_cluster_distances(X, labels, centroids, lattice_vectors):
            """Calculate distances between points and cluster centers considering PBCs."""
            distances = {}
            for i in range(len(centroids)):
                cluster_points = X[labels == i]
                distances[i] = []
                for point in cluster_points:
                    distance = np.round(periodic_distance(point, centroids[i], lattice_vectors), decimals=2)
                    distances[i].append(distance)
            return distances

        def calculate_min_inter_centroid_distances(centroids, lattice_vectors):
            """Calculate the nearest-neighbor centroid distance for each cluster under PBCs."""
            min_distances = []
            for i in range(len(centroids)):
                distances = np.array([periodic_distance(centroid, centroids[i], lattice_vectors) for centroid in centroids])
                min_distance = np.min(distances[distances > 0])
                min_distances.append(min_distance)
            return np.round(min_distances, decimals=2)

        labels = {}
        centroids_cartesian = {}
        sites_cluster = {}
        cluster_min = {}
        cluster_radiis = {}
        for wp, coord in self.coords.items():
            atomic_positions = self.fractional_to_cartesian(coord)

            optimal_k = _find_optimal_k(
                atomic_positions,
                random_state=self.random_state,
            )

            periodic_kmeans = KMeans(
                n_clusters=optimal_k,
                n_init=self.n_init,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
            ).fit(atomic_positions)

            labels[wp] = periodic_kmeans.labels_
            centroids_cartesian[wp] = periodic_kmeans.cluster_centers_

            sites_cluster[wp] = calculate_intra_cluster_distances(atomic_positions, labels[wp], centroids_cartesian[wp], self.cell)
            cluster_min[wp] = calculate_min_inter_centroid_distances(centroids_cartesian[wp], self.cell)

            if plot:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                for i in range(periodic_kmeans.n_clusters):
                    cluster_points = atomic_positions[labels[wp] == i]
                    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f"Cluster {i}")
                ax.scatter(centroids_cartesian[wp][:, 0], centroids_cartesian[wp][:, 1], centroids_cartesian[wp][:, 2], marker='x', color='red', s=100, label="Centroids")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.title("Periodic K-Means Clustering of Atomic Positions under PBC")
                plt.legend()
                plt.show()

            cluster_radiis[wp] = []
            for key, val in sites_cluster[wp].items():
                if np.mean(val) <= 0.5:
                    cluster_radiis[wp].append(np.round(np.mean(val) + 1.5, decimals=2))
                else:
                    cluster_radiis[wp].append(np.round(np.mean(val) + 0.5, decimals=2))
        return centroids_cartesian, labels, sites_cluster, cluster_min, cluster_radiis

    def visualize(self, wp: str) -> go.Figure:
        """Plot clusters through plotly."""

        centroids_cartesian, labels, sites_cluster, cluster_min, sphere = self.KClassCenters(plot=False)

        def visualize_clusters_3d(X, labels, centroids, max_distance):
            """Visualize clusters in 3D with spheres centered at cluster centers."""
            fig = go.Figure()

            unique_labels = np.unique(labels)
            for label in unique_labels:
                subset = X[labels == label]
                trace = go.Scatter3d(
                    x=subset[:, 0],
                    y=subset[:, 1],
                    z=subset[:, 2],
                    mode='markers',
                    marker=dict(size=5, opacity=0.8),
                    name=f'LCS-{label+1}',
                )
                fig.add_trace(trace)

            centroid_trace = go.Scatter3d(
                x=centroids[:, 0],
                y=centroids[:, 1],
                z=centroids[:, 2],
                mode='markers',
                marker=dict(size=10, opacity=1, color='red'),
                name='Centroids'
            )
            fig.add_trace(centroid_trace)

            for center, rad in zip(centroids, max_distance):
                u = np.linspace(0, 2 * np.pi, 50)
                v = np.linspace(0, np.pi, 50)
                x_sphere = rad * np.outer(np.cos(u), np.sin(v)) + center[0]
                y_sphere = rad * np.outer(np.sin(u), np.sin(v)) + center[1]
                z_sphere = rad * np.outer(np.ones_like(u), np.cos(v)) + center[2]

                sphere_trace = go.Surface(
                    x=x_sphere,
                    y=y_sphere,
                    z=z_sphere,
                    opacity=0.1,
                    showlegend=False,
                    coloraxis=None,
                    colorscale='Blues',
                    showscale=False
                )
                fig.add_trace(sphere_trace)

            cell = self.cell
            vertices = [
                np.array([0, 0, 0]), cell[0], cell[1], cell[2],
                cell[0] + cell[1] + cell[2], cell[1] + cell[2], cell[0] + cell[2], cell[0] + cell[1],
                cell[0], cell[0] + cell[2], np.array([0, 0, 0]), cell[1] + cell[0],
                cell[1] + cell[2], cell[1], cell[2], cell[1] + cell[2] + cell[0],
                cell[2], cell[0] + cell[2],
                cell[0] + cell[1], cell[1]
            ]

            edges_all = [
                [[0, 1], [0, 2], [0, 3]],
                [[4, 5], [4, 6], [4, 7]],
                [[8, 9], [8, 10], [8, 11]],
                [[12, 13], [12, 14], [12, 15]],
                [[16, 17]],
                [[18, 19]]
            ]

            for edges in edges_all:
                fig.add_scatter3d(
                    x=[vertices[i][0] for edge in edges for i in edge],
                    y=[vertices[i][1] for edge in edges for i in edge],
                    z=[vertices[i][2] for edge in edges for i in edge],
                    mode='lines',
                    line=dict(color='black', width=3),
                    showlegend=False
                )

            fig.update_layout(
                scene=dict(
                    xaxis=dict(showgrid=False, showbackground=False, backgroundcolor=None, visible=True),
                    yaxis=dict(showgrid=False, showbackground=False, color=None, visible=True),
                    zaxis=dict(showgrid=False, showbackground=False, backgroundcolor=None, visible=True),
                    xaxis_title='X-axis (Å)',
                    yaxis_title='Y-axis (Å)',
                    zaxis_title='Z-axis (Å)',
                ),
                width=800,
                height=800,
                font=dict(family="Arial", size=18, color="black")
            )
            fig.show()
            return fig

        return visualize_clusters_3d(self.fractional_to_cartesian(self.coords[wp]), labels[wp], centroids_cartesian[wp], max_distance=sphere[wp])


class SuperCell(PBCKmeans):
    """Supercell construction from clustered unit cell coordinates."""

    def __init__(self, coords: dict[str, np.ndarray], cell: np.ndarray) -> None:
        super().__init__(coords=coords, cell=cell)

    @staticmethod
    def _compute_scaling_factor(scaling_mat: np.ndarray) -> int:
        """Compute the integer scaling factor from a scaling matrix."""

        return int(np.sqrt(
            np.sum(scaling_mat[0] ** 2)
            * np.sum(scaling_mat[1] ** 2)
            * np.sum(scaling_mat[2] ** 2)
        ))

    def transform_to_supercell(self, cluster_coord: np.ndarray, scaling_mat: np.ndarray, frac: bool = False) -> np.ndarray:
        """Transform unit cell sites to supercell with scaling matrix."""

        unit_cell_coords = np.array(cluster_coord)
        struct = Structure(self.cell, ["Li"] * len(cluster_coord), unit_cell_coords, coords_are_cartesian=True)
        if not frac:
            supercell_coords = np.dot(struct.make_supercell(scaling_mat).frac_coords, struct.lattice.matrix)
        else:
            supercell_coords = struct.make_supercell(scaling_mat).frac_coords
        return supercell_coords

    def KClassCenterSuper(
        self,
        scaling_mat: np.ndarray,
        clusters: dict[str, np.ndarray],
        plot: bool = False,
        frac: bool = False,
    ) -> dict[str, list[np.ndarray]]:
        """Return cluster centers in supercell with scaling matrix."""

        clusters_super: dict[str, list[np.ndarray]] = {}
        for wp, centers in clusters.items():
            clusters_super[wp] = []
            for coord in centers:
                cart_centers = self.transform_to_supercell(np.array([coord]), scaling_mat, frac=frac)
                clusters_super[wp].append(cart_centers)

        if plot:
            scaling_factor = self._compute_scaling_factor(scaling_mat)
            for wp, centers in clusters_super.items():
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                for i in range(len(clusters[wp])):
                    ax.scatter(*clusters_super[wp][i].T, label=f'Cluster {i+1}')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.legend()
                plt.show()
        return clusters_super

    def KClassCoordsSuper(
        self,
        scaling_mat: np.ndarray,
        coords: dict,
        label: dict,
        plot: bool = False,
        frac: bool = False,
    ) -> tuple[dict, dict]:
        """Return labeled coordinates in supercell with scaling matrix."""
        coords_with_labels: dict = {}

        wp_keys = label.keys()
        for wp in wp_keys:
            coords_with_labels[wp] = {}
            for coord, idx in zip(coords[wp], label[wp]):
                coord = self.fractional_to_cartesian(coord)
                if coords_with_labels[wp].get(idx):
                    coords_with_labels[wp][idx].append(coord)
                else:
                    coords_with_labels[wp][idx] = []
                    coords_with_labels[wp][idx].append(coord)

        super_coords: dict = {}
        for wp in wp_keys:
            super_coords[wp] = {}
            for idx, coord in coords_with_labels[wp].items():
                final_coords = self.transform_to_supercell(np.array(coord), scaling_mat, frac=frac)
                if super_coords[wp].get(idx):
                    super_coords[wp][idx].append(final_coords)
                else:
                    super_coords[wp][idx] = []
                    super_coords[wp][idx] = final_coords

        scaling_factor = self._compute_scaling_factor(scaling_mat)
        for wp in wp_keys:
            for idx, coord in super_coords[wp].items():
                num_li = int(np.array(coord).shape[0] / scaling_factor)
                reshaped = coord.reshape(num_li, scaling_factor, 3)
                reshaped_clusters = np.array([[reshaped[lithium][repeat]]
                                              for repeat in range(scaling_factor)
                                              for lithium in range(num_li)]).reshape(scaling_factor, num_li, 3)
                super_coords[wp][idx] = reshaped_clusters

        if plot:
            for wp, centers in super_coords.items():
                num_clusters = len(coords_with_labels[wp])

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                for idx, coord_arr in centers.items():
                    for r in range(scaling_factor):
                        ax.scatter(*coord_arr[r].T, label=f'Cluster {idx+1}' if r == 0 else None)

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.legend()
                plt.show()
        return super_coords, coords_with_labels
