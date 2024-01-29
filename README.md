# CarsiInduce

## Project Description

CarsiInduce: an equivariant deep learning-based model designed to refine ESMFold-predicted protein pockets with the induction of specific ligands. 

## Setup Environment
This is an example for how to set up a working conda environment to run the code (but make sure to use the correct pytorch, pytorch-geometric, cuda versions or cpu only versions):
```bash
conda create --name CarsiInduce python=3.9
conda activate CarsiInduce
conda install pytorch==1.11.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==2.0.4 -f https://data.pyg.org/whl/torch-1.11.0+cu117.html
python -m pip install PyYAML scipy "networkx[default]" biopython rdkit-pypi e3nn spyrmsd pandas biopandas
```

## Data and Models
- Data：`./example_data/posebusters_esmfold`
    - download location : https://zenodo.org/records/10575464

- Model weights：`./ckpt/step59000_model.pt`
    - download location : https://zenodo.org/records/10575464

## inference command
```bash
python inference_esmfold.py 
```

## The induction effect of the protein was evaluated
```bash
python compute_pocket_rmsd.py
```

## CarsiDock-Flex = CarsiInduce + CarsiDock, using induced pocket to docking ligand
### CarsiInduce
### CarsiDock
## License
The code of this repository is licensed under [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.en.html). The use of the CarsiInduce model weights is subject to the [Model License](./MODEL_LICENSE.txt). CarsiInduce weights are completely open for academic research, please contact <bd@carbonsilicon.ai> for commercial use. 

## Copyright
[CarbonSilicon.AI](https://carbonsilicon.ai/) All rights reserved.
