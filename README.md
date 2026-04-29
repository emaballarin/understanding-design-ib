# Understanding and inverse design of implicit bias in stochastic learning: a geometric perspective

Source code for experiments and figures described in the paper _Understanding and inverse design of implicit bias in stochastic learning: a geometric perspective_ (Aladrah et al., 2026).

## Repository structure

The `src` folder contains source code responsible for data generation, data analysis, and plotting. In particular:

- File `01_shallowrelu.py` is the script to be run for the _Shallow ReLU network_ experiment described in Fig. 2.
- File `02_attention.py` is the script to be run for the _Single-Head SDPA_ experiment described in Fig. 3.
- File `01_suppl_rank_matcomp.py` is the script to be run for the _Low-rank matrix completion_ experiment described in Fig. 4.
- File `03_spectral_sparse.py` is the script to be run for the _Sparse spectral recovery_ experiment described in Fig. 5.
- File `04_tv_regularization.py` is the ecript to be run for the _Piecewise-constant signal recovery_ experiment described in Fig. 6.
- File `00_redo_plots.py` is the script to be run for the generation of paper plots from the data saved by individual experiments.

The `saved` folder contains pre-generated [`safetensors`](https://huggingface.co/docs/safetensors/index) files in which the results of individual experiments are stored, allowing figure generation without running the actual set of experiments.

The `figures` folder contains PNG versions of the figures shown in the paper.

## System requirements

The code has been developed for, and tested on, Linux systems. The only strict requirement for full reproduction is the availability of GLibC `>=v2.28` (required by of PyTorch), which can be assumed to be satisfied on any sufficiently recent and updated Linux distribution. In particular, the code was tested on ArchLinux `>=2026.03.01` and Rocky Linux `9.5` running on x86_64 processors.

Specific software requirements are listed in file `pyproject.toml` and can be installed using [uv](https://docs.astral.sh/uv/) and a working Internet connection (see below for further instructions). In detail:

| Package          | Version     |
| ---------------- | ----------- |
| Python           | `>= 3.14`   |
| `torch`          | `>= 2.11`   |
| `numpy`          | `>= 2.4.4`  |
| `matplotlib`     | `>= 3.10.7` |
| `safetensors`    | `>= 0.7`    |
| `simple-parsing` | `>= 0.1.8`  |
| `tqdm`           | `>= 4.67.3` |

A working LaTeX installation is required for proper typesetting of figure labels by Matplotlib.

### Installation guide

The easiest way to install all required dependencies to reproduce the experiments is to install the `uv` package manager, by following the [official documentation](https://docs.astral.sh/uv/#installation), or by simply running in a user shell:

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After that, from the repository root, one can invoke

```shell
uv sync
```

to automatically install required dependencies from `pyproject.toml`. The Python interpreter will be located at `.venv/bin/python`.

Install time will be strongly dependent on Internet connection speed. On reasonably fast academic networks it should not take more than 5 minutes.

## Reproducibility instructions

In order to reproduce the experiments contained in the paper, one should open a user shell within the `src` folder and then run with a suitable Python interpreter (e.g. the one prepared earlier) the script(s) for the experiment(s) of interest. E.g., for full reproduction, one can run:

```shell
python -O 01_shallowrelu.py
python -O 01_suppl_rank_matcomp.py
python -O 02_attention.py
python -O 03_spectral_sparse.py
python -O 04_tv_regularization.py
```

Each invoked script will generate algorithmically the data required by the experiment, train the associated models, and report relevant results (e.g. those reported in figure captions). Diagnostic plots, and data to re-generate them without running the full experiments, will also be generated and saved on disk.

To replicate publication-quality figures, the dedicated script can be run, e.g.:

```shell
python -O 00_redo_plots.py
```

According to CPU capabilities, the expected runtime for the full experiment battery in between 1h and 2h.

## License

[MIT](LICENSE)

## Citation

```bibtex
@misc{aladrah2026implicit,
    title         = {Understanding and inverse design of implicit bias in stochastic learning: a geometric perspective},
    author        = {Aladrah, Nicola and Ballarin, Emanuele and Biagetti, Matteo and Ansuini, Alessio and d'Onofrio, Alberto and Anselmi, Fabio},
    year          = {2026},
    eprint        = {2601.06597},
    archivePrefix = {arXiv},
    primaryClass  = {cs.LG}
}
```
