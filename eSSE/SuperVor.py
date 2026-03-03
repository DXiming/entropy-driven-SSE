"""
Voronoi tessellation of clusters in SSEs.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pyvoro


class VoronoiCenter():
    """Discretize MD trajectories based on Voronoi algorithm."""

    def __init__(self, super_coords: dict, super_centers: dict) -> None:
        super(VoronoiCenter, self).__init__()
        self.super_coords = super_coords
        self.super_centers = super_centers

    def get_vors(self, limits: float = 0.5) -> dict:
        """Get Voronoi cells of clusters.

        Args:
            limits: Padding added to the bounding box of each cluster.

        Returns:
            Nested dict {wp: {center: {'points', 'limits', 'voronoi', 'center_pos'}}}.
        """
        vors: dict = {}
        for wp in self.super_coords.keys():
            vors[wp] = {}
            for center in self.super_coords[wp].keys():
                vors[wp][center] = {}
                vors[wp][center]['points'] = self.super_coords[wp][center]
                cell_size = self.super_coords[wp][center].shape[0]
                vors[wp][center]['limits'] = np.array([[x.min() - limits, x.max() + limits]
                                                              for d in range(len(self.super_coords[wp][center]))
                                                              for x in self.super_coords[wp][center][d].T]).reshape(cell_size,
                                                                                                                    3,
                                                                                                                    2)
                vors[wp][center]['voronoi'] = [pyvoro.compute_voronoi(points,
                                                                     limits,
                                                                     1,
                                                                     periodic=[True, True, True]
                                                                     ) for points, limits in zip(vors[wp][center]['points'],
                                                                                                 vors[wp][center]['limits'])]
                vors[wp][center]['center_pos'] = self.super_centers[wp][center]
        return vors

    def vis_vors(self, wp: str, limits: float) -> go.Figure:
        """Plot Voronoi cells for visualization.

        Args:
            wp: Wyckoff site label to visualize.
            limits: Padding for the Voronoi bounding box.

        Returns:
            Plotly figure with 3D Voronoi cell visualization.
        """
        fig = go.Figure()
        vors = self.get_vors(limits)

        for center in self.super_coords[wp].keys():
            for i in range(len(vors[wp][center]['voronoi'])):
                for part in range(len(vors[wp][center]['voronoi'][i])):
                    for poly in vors[wp][center]['voronoi'][i][part]:
                        verts = vors[wp][center]['voronoi'][i][part]["vertices"]
                        color_list = [
                            "#2c1bc5", "#77f39c", "#ec8739",
                            "#b863e0"
                        ]
                        color = color_list[list(self.super_coords[wp].keys()).index(center) % len(color_list)]
                        fig.add_trace(go.Mesh3d(
                            x=np.array(verts)[:, 0],
                            y=np.array(verts)[:, 1],
                            z=np.array(verts)[:, 2],
                            alphahull=0.1,
                            opacity=0.02,
                            color=color,
                        ))
            points = vors[wp][center]['points']
            num_points = points.shape[0] * points.shape[1]
            points = points.reshape(num_points, 3)
            point_trace = go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=6,
                    opacity=1,
                    color=color,
                ),
                name=f'Points in center {center}'
            )
            fig.add_trace(point_trace)

        fig.update_layout(
            autosize=False,
            height=800,
            width=800,
        )
        fig.show()
        return fig
