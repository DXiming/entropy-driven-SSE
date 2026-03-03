# **e**ntropy-driven **S**olid-**S**tate **E**lectrolyte (**eSSE**) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

`eSSE` is a Python package for **path entropy analysis** of solid-state electrolytes (SSEs). It provides tools to quantitatively assess ionic diffusion by analysing the diversity of ion migration pathways, with a focus on lithium-ion transport kinetics.

![Path entropy illustration](images/path_entropy.png)

## Table of Contents

- [Related Work](#related-work)
- [Prerequisites](#prerequisites)
- [Data](#data)
- [Demo](#demo)
- [Author](#author)

---

## Related Work

If you find this repository useful in your research, please consider citing our [related work](https://arxiv.org/abs/2412.07115):

```bibtex
@misc{guan2025pathentropydrivendesignsolidstate,
      title={Path Entropy-driven Design of Solid-State Electrolytes}, 
      author={Qiye Guan and Kaiyang Wang and Jingjie Yeo and Yongqing Cai},
      year={2025},
      eprint={2412.07115},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2412.07115}, 
}
```

---

## Prerequisites

`eSSE` requires **Python 3.10**. The recommended approach is to use [conda](https://conda.io/docs/index.html) to manage your environment.

**1. Create and activate a new conda environment:**

```bash
conda create -n your_env_name python=3.10.16
conda activate your_env_name
```

**2. Clone this repository:**

```bash
git clone https://github.com/DXiming/entropy-driven-SSE.git
cd entropy-driven-SSE
```

**3. Install `eSSE` in editable mode:**

```bash
pip install -e .
```

This will automatically install all required dependencies as specified in `pyproject.toml`.

---

## Data

Precomputed discrete trajectories (`data/trajs/disc_trajs.pkl.gz`) are included so that path-entropy analysis in `examples/demo.ipynb` can be run directly. To reproduce the full workflow from raw MD trajectories, download trajectories from the [Zenodo repository](https://doi.org/10.5281/zenodo.18829656).

---

## Demo

The easiest way to get started is to run the provided notebook [examples/demo.ipynb](./examples/demo.ipynb)

It walks through the full path entropy analysis workflow step by step. Open the notebook and run all cells after completing the installation steps above:

```bash
cd examples
jupyter notebook demo.ipynb
```

---

## Author

This package was developed and is maintained by **Qiye Guan**.
