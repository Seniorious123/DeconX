# Installation

DeconX requires **Python 3.10+** and a CUDA-capable GPU is recommended (the
training stage uses PyTorch). It has been verified on a single MIG slice of an
NVIDIA A100 (1 g.5 gb partition, 5 GB VRAM) — see
[Hyperparameters](user_guide/hyperparameters.md) for runtime details.

## From source (recommended)

```bash
git clone https://github.com/Seniorious123/DeconX.git
cd DeconX
pip install -r requirements.txt
```

## Core dependencies

| Category       | Packages                                            |
|----------------|-----------------------------------------------------|
| Deep learning  | `torch`                                             |
| Single-cell    | `scanpy`, `anndata`                                 |
| Data science   | `numpy`, `pandas`, `scipy`                          |
| Visualisation  | `matplotlib`, `seaborn`                             |
| Utilities      | `tqdm`                                              |

A minimal install command:

```bash
pip install torch scanpy anndata pandas numpy matplotlib seaborn tqdm scipy
```

## GPU vs CPU

DeconX runs on either, but a 5 GB MIG slice (or any modern consumer GPU) is more
than enough — peak GPU memory across all benchmarked runs (n = 100 to 5000
pseudobulk samples) was 215–574 MB.

If no GPU is available, set the environment variable `CUDA_VISIBLE_DEVICES=""`
or pass `DEVICE='cpu'` in your run script. CPU runs are 5–10× slower.

## Verifying the install

After installation, run the simulation entry point with the small default
sample size:

```bash
python scripts/run_simulation.py --num-samples 100 --experiment-name smoke_test
```

A successful run produces `outputs/simulated_data/smoke_test/{bulk.csv,frac.csv}`.
