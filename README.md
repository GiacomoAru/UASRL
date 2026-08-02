# UASRL

Master Thesis Extension.

## Python environment

The project environment is tested with Python 3.10.11. From PowerShell in this
directory, create and populate the virtual environment with:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The requirements include the CUDA 12.4 build of PyTorch. A compatible NVIDIA
driver is required for GPU execution; PyTorch can still be imported and used on
the CPU when no compatible GPU is available.

Verify the installation and the CBF implementation with:

```powershell
.\.venv\Scripts\python.exe -c "import torch, mlagents, osqp; print(torch.__version__)"
.\.venv\Scripts\python.exe -m unittest tests.test_cbf -v
```
