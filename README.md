# Iterated EqF landmark simulation

This repository contains the simulation used to compare a one-step EqF with an
iterated EqF for planar localisation on \(SE(2)\). The vehicle follows a
tangent-aligned figure-eight Lissajous trajectory and receives noisy range
measurements from two landmarks, with an optional third landmark.

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib
- A working LaTeX installation for Matplotlib text rendering with LaTeX

Install the Python dependencies with:

```bash
python -m pip install numpy matplotlib
```

## Run the experiment

The default command performs 100 Monte Carlo trials and saves the figure as
`iterated_eqf_statistics.pdf`:

```bash
python iterated_eqf_landmarks.py
```

Useful options:

```bash
python iterated_eqf_landmarks.py --mc 200 --seed 7
python iterated_eqf_landmarks.py --third-landmark
python iterated_eqf_landmarks.py --save results.pdf --show
```

Use `python iterated_eqf_landmarks.py --help` for the complete command-line
interface.

## Publication

This code was produced as part of an academic paper.
If you use this code in your research, please cite:

```bibtex
@article{vanGoor2026iterative,
  author  = {van Goor, Pieter and Forbes, James Richard},
  title   = {The Iterative Equivariant Filter},
  journal = {arXiv preprint arXiv:2609.12328},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.12328}
}
```

To reproduce the figure featured in the paper, use the command:

```bash
python iterated_eqf_landmarks.py --mc 100 --third-landmark --nees
```

## License

This software is licensed under the GNU General Public License,
version 3 or later. See [LICENSE](LICENSE) for the full license text.
