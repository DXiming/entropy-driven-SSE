from __future__ import annotations

import warnings

import numpy as np

from .SSEMarkov import MarkovSSE, PostAna

warnings.filterwarnings('ignore')


def analyze_msm(
    wp: str,
    disc_trajs: dict[int, list[np.ndarray]],
    states_with_labels: dict,
    lagtime: int = 600,
    length_step: float = 20,
) -> dict:
    """Run MSM post-analysis for every cluster center of a Wyckoff site.

    Args:
        wp: Wyckoff site label.
        disc_trajs: Discrete trajectories grouped by cluster center.
        states_with_labels: State-label mapping from ``DiscreteTraj.get_state_labels()``.
        lagtime: Lag time for transition count estimation.
        length_step: Physical time per step (used to scale MFPT).

    Returns:
        Nested dict of MSM post-analysis results.
    """
    msm_info: dict = {}
    msm_info[wp] = {}

    for center in list(states_with_labels[wp].keys()):
        msm_info[wp][center] = []
        for disc_traj in disc_trajs[center]:
            msm_info[wp][center].append(MarkovSSE(disc_traj).post_analysis(lagtime=lagtime, length_step=length_step))

    return msm_info


def analyze_path_entropy(wp, msm_info, states_with_labels, PA, escape_entropy=False,partial=False, counts_ratio=0.15):
    """Path entropy analysis for given MSM results and parameters.
    
    Args:
        wp: Wyckoff site label.
        msm_info: MSM post-analysis results.
        states_with_labels: State-label mapping from ``DiscreteTraj.get_state_labels()``.
        PA: PostAna object.
        escape_entropy: If True, calculate escape entropy.
        partial: If True, restrict to the local LCS.
        counts_ratio: Counts ratios for entropy calculation.

    Returns:
        Final entropy, standard error of the mean
    """
    ps_all = {}
    for center in states_with_labels[wp].keys():
        if escape_entropy:
            entropy = PA.escape_entropy(results=msm_info, 
                            wp=wp,
                            center=center,
                            intermediates=4,
                            flux_type="tpt_net",    
                            counts_ratio=counts_ratio,
                            partial=partial)
            print(f"S_e (center {center}) = {entropy:.2f} J/mol/K")
        else:
            entropy = PA.path_entropy(results=msm_info, 
                            wp=wp,
                            center=center,
                            intermediates=4,
                            flux_type="tpt_net", 
                            counts_ratio=counts_ratio,
                            partial=partial)
            print(f"S_p (center {center}) = {entropy:.2f} J/mol/K")
        ps_all[center] = entropy
    return ps_all
