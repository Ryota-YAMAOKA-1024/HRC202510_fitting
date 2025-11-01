# FeS Spin-Wave Fitting Toolkit

This repository bundles the working scripts, configuration files, and helper assets we currently use in the FeS joint project to fit inelastic neutron-scattering data. The core workflow centres on a Python fitter (`fitting_FeS.py`) that ingests experimental peak positions, optimises the exchange/anisotropy parameters of the spin-wave model, and produces publication-ready diagnostics (CSV summaries and PNG overlays). A supplementary Gnuplot script (`peak/fitting.gp`) is included for legacy peak-by-peak cross checks.

The repository is intended for collaborators. The sections below describe how the code is structured, how to prepare inputs, how to run the fit, and how to customise the outputs for your datasets.

---

## Contents

- `fitting_FeS.py` – Main fitting driver written in Python. Handles
  - reading experimental points from `list.csv` or `list.txt`;
  - reading parameter bounds/locks from `initial_list.txt`;
  - bounded non-linear least-squares (SciPy if available, otherwise internal LM fallback);
  - residual logging and figure generation (`H00_E.png`, `K-K0_E.png`, `00L_E.png`);
  - export of best-fit parameters to `fit_results.csv`.
- `initial_list.txt` – Human-readable parameter configuration (initial values, lock/free flag, lower/upper bounds for each parameter `J1`, `J2`, `J3`, `J_alt`, `J_prime_alt`, `S`, `D`).
- `list.csv` / `list.txt` – Experimental data (`H,K,L,E`). Any line beginning with `#` or empty lines are ignored.
- `fit_results.csv` – Automatically generated summary of the most recent fit (parameter estimates, bounds, success flag, residual sum of squares, initial guess used).
- `H00_E.png`, `K-K0_E.png`, `00L_E.png` – Automatically generated figures comparing experimental points and fitted dispersion along the principal scan directions.
- `peak/fitting.gp` – Optional Gnuplot recipe for single-peak fitting (kept for backward compatibility).
- `trash/` – Scratch directory (ignored by the fitter).

---

## Requirements

- Python ≥ 3.10 (tested with the macOS Homebrew build of Python 3.13).
- Mandatory Python packages: `numpy`, `matplotlib`.
- Optional (recommended) package: `scipy` for its robust `least_squares` implementation. When SciPy is not available, the script falls back to an internal Levenberg–Marquardt routine.
- Gnuplot (only required if you intend to run `peak/fitting.gp`).

Use your existing virtual environment or create a dedicated one:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scipy
```

---

## Preparing Inputs

### Experimental points (`list.csv` or `list.txt`)
- Format: comma-separated with four numeric columns `H,K,L,E`.
- Comment lines beginning with `#` and blank lines are ignored.
- All rows must provide a valid energy value (`E`); empty fields trigger a parsing error.
- Place the file in the same directory as `fitting_FeS.py`. The script automatically prioritises `list.csv`, then `list.txt`.

### Parameter configuration (`initial_list.txt`)
- Each non-comment line has the form:

  ```
  parameter, initial_value, status(lock|free), lower_bound, upper_bound
  ```

- Use `lock` to freeze a parameter at `initial_value`; bounds are optional in this case.
- Use `free` to allow fitting; both `lower_bound` and `upper_bound` are required.
- All seven parameters `J1`, `J2`, `J3`, `J_alt`, `J_prime_alt`, `S`, and `D` must be present exactly once.
- The fitter transforms free parameters internally to honour bounds (via a logistic mapping), so choose realistic limits that reflect expected physics.

---

## Running the Fit

From the repository root (`code/`):

```bash
python3 fitting_FeS.py
```

The script will:

1. Load data (`list.csv` or `list.txt`) and settings (`initial_list.txt`).
2. Build the initial internal parameter vector (respecting locked parameters).
3. Launch the optimiser with the user-specified guess plus up to `RANDOM_GUESS_COUNT` additional random initial conditions sampled within the bounds.
4. Select the solution with the lowest residual sum of squares.
5. Save results to `fit_results.csv` and generate the three diagnostic PNGs.

The command exits with a descriptive error message if input files are missing, ill-formatted, or if the optimisation fails to converge.

### Command-line options

`fitting_FeS.py` currently takes no command-line arguments; customise behaviour via the configuration constants at the top of the script:

- `RANDOM_GUESS_COUNT`, `RANDOM_SEED` – number of random restarts and their RNG seed.
- `PLOT_ASPECT` – aspect ratio per axis (`None` for auto scaling, or any value accepted by `matplotlib.axes.Axes.set_aspect`).
- `LM_MAX_ITER` – maximum iterations for the fallback solver when SciPy is unavailable.

---

## Outputs

### `fit_results.csv`
| column        | description                                                                 |
|---------------|------------------------------------------------------------------------------|
| parameter     | Name of the fitted/locked parameter.                                        |
| value         | Best-fit value (or locked value) in the model units.                        |
| status        | `free` or `locked`, reflecting `initial_list.txt`.                          |
| bounds        | Interval enforced during fitting. Locked parameters are reported `[v, v]`.  |
| residual_sum  | (Single row after the table) Sum of squared residuals at the optimum.       |
| success       | Boolean flag from the optimiser (`True` if convergence criteria satisfied). |
| message       | Convergence message (SciPy output or fallback solver status).              |
| initial_guess | Physical-space initial guess that produced the best solution.               |

### Figures
- `H00_E.png`, `K-K0_E.png`, `00L_E.png` show the experimental points (blue markers) and the continuous fitted curve (orange line). The L-scan uses the hard-coded range `(3, 4)`; adjust the `plot_axis()` calls if your experimental window differs.
- Y-axis starts at zero by default. Modify `ax.set_ylim()` inside `plot_axis()` to relax this.

---

## Customisation Tips

- **Changing aspect ratios**: edit the `PLOT_ASPECT` dictionary near the top of `fitting_FeS.py`. Example: `PLOT_ASPECT["H"] = 'equal'`.
- **Tweaking random initial guesses**: adjust `RANDOM_GUESS_COUNT` or the sampling ranges in `sample_random_internal()`.
- **Forcing parameters to remain positive**: set the lower bound accordingly in `initial_list.txt`.
- **Alternative data windows**: update the `plot_axis()` calls at the bottom of `generate_plots()` to change the x-range used for plotting.

---

## Legacy Gnuplot Fitting (`peak/fitting.gp`)

For collaborators who prefer single-peak Gnuplot fits:

```gnuplot
gnuplot> load "peak/fitting.gp"
```

`fitting.gp` expects a two-column text file (intensity vs. momentum) with errors as the third column. Update the `datafile` variable at the top of the script, provide sensible initial guesses (`BG`, `A1`, `A2`, `q1`, `q2`, `w`, `d`), and the script writes fit diagnostics to `fitlog.txt` and a PNG overlay (`plot_*.png`).

---

## Best Practices & Troubleshooting

- **Missing SciPy warning**: install SciPy for improved robustness (`pip install scipy`). The fallback solver works but may require more iterations.
- **Parsing errors**: if the script complains about float conversion, inspect the offending line in `list.csv` for missing energy values.
- **Convergence issues**: relax parameter bounds, supply a better initial guess, or increase `RANDOM_GUESS_COUNT`.
- **Matplotlib cache warning**: if running on macOS without write access to the default cache, set `MPLCONFIGDIR=/tmp` before running the script (e.g., `MPLCONFIGDIR=/tmp python3 fitting_FeS.py`).

---

## Version Control & Collaboration

When preparing changes for GitHub:

1. Update `initial_list.txt` and `list.csv` with anonymised or shareable values.
2. Run the fitter to ensure reproducibility (`fit_results.csv` and PNGs will be regenerated).
3. Commit the relevant artefacts (typically the Python script, configuration, and README; large intermediate files can remain untracked).

Please add run notes or parameter justifications in your commit messages or within the README to keep collaborators informed of modelling assumptions.

---

Happy fitting!
