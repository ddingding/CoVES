# CoVES
Structure-based scoring and sampling of 'Combinatorial Variant Effects from Structure' (CoVES) 

For the accompanying paper see: https://www.biorxiv.org/content/10.1101/2022.10.31.514613v2

## Folder structure
The respective folders contain examples of how to run the code.
- ./data/ contains various required data sources of experimental variant effect measurements, PDB structures
- ./supervised_regression/ contains scripts to perform nonlinear, residue-mutation preference regression on various experimental datasets
- ./coves/ contains notebooks for calculating, sampling and scoring from structural-microenvironment based residue preferences
- ./src/ contains helper functions for analysis
- ./other_tools/ contains analysis pipeline for scoring and sampling from ESM-IF, EvCouplings and Protein-MPNN

## Requirements
- Tensorflow 2.0
- Python 3.9
- Torch 1.9.0
- if using, see the respective packages (ESM-IF, EvCoupings, ProteinMPNN) for requirements

Note: Some scripts are tied to our research compute cluster (Harvard O2 research computing with Slurm Workload Manager), and are provided as reference only.

Please feel free to contact (davidding at berkeley dot edu) or (ayshaw at g dot harvard dot edu) for questions.
