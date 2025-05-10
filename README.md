# RustConv
Project for Emerging Topics in CS Course – LLMs for HPC Code Gen – CS 5914

### Setup Instructions
To replicate our results:
- Download or Clone this repository.
- Create a Conda environment and activate it.
- Install the required dependencies (Pytorch, Pillow, Altair, Tqdm, etc...).
- Navigate to the `./Python` folder.
- From the terminal, execute `python execute_experiments.py`
- (Optional) To visualize the data, open and run the `analyze_experiments.ipynb` Jupyter Notebook

⚠️ Note: The current binaries in `./Python/executables` were compiled for MacOS with Apple Silicon. To execute the experiments on other architectures, the rust projects must be recompiled for the target architecture using Cargo's `--release` flag.