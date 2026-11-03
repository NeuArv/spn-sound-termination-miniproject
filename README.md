# SPN model – setup

Single-compartment SPN neuron model from Kopp-Scheinpflug et al. (2011), *Neuron* 71:911–925
(ModelDB [139657](https://modeldb.science/139657)).

## Requirements

- Python 3.9+ (must be a version supported by your NEURON release)
- [NEURON](https://nrn.readthedocs.io/en/latest/install/install_instructions.html) 8 or newer (tested with 9.0.2)
- numpy, matplotlib (plus jupyter and neurom for the notebook)

## 1. Install NEURON

**Windows:** pip does not provide NEURON on Windows. Download and run the installer (`.exe`) from
<https://github.com/neuronsimulator/nrn/releases>. Keep the options that add NEURON to `PATH` and
set up Python. Then open a new terminal.

**macOS / Linux:**

```bash
pip install neuron
```

Check that it works:

```bash
python -c "import neuron; print(neuron.__version__)"
```

## 2. Install the Python packages

```bash
pip install numpy matplotlib jupyter neurom
```

`neurom` is only needed because `spn_model.ipynb` imports it on its first line. That import is not
used, so you can delete it instead of installing `neurom`.

## 3. Compile the mechanisms (`.mod` files)

Run this inside the `simulationcode` folder:

| OS | Command | Result |
|----|---------|--------|
| Windows | `nrnivmodl` (or `mknrndll`) | `nrnmech.dll` |
| macOS / Linux | `nrnivmodl` | `x86_64/` or `arm64/` folder |

NEURON loads the compiled mechanisms automatically when Python is started from this folder.
A prebuilt `nrnmech.dll` (Windows, NEURON 9.0.2) is included. If it fails to load, compile again.
Recompile after upgrading NEURON or editing a `.mod` file. Delete the old `*.o`, `*.cpp` and
`nrnmech.dll` first.

## Troubleshooting

- `AttributeError: ... has no attribute 'sjg_ih'` (or `mhh_Gfluct`, `NetStims`, ...): the mechanisms
  are not loaded. Run Python from the `simulationcode` folder, or compile them again (step 3).
- `ModuleNotFoundError: No module named 'neuron'` on Windows: NEURON's Python path was not set up.
  Reinstall NEURON with the Python option, or add `C:\nrn\lib\python` to `PYTHONPATH`.
