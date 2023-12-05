
# CoVES
Structure-based scoring and sampling of 'Combinatorial Variant Effects from Structure' (CoVES) 

For the accompanying paper see: https://www.biorxiv.org/content/10.1101/2022.10.31.514613v2

## Folder Structure & Usage
The respective folders contain example notebooks of how to perform a given analysis.

### Supervised fitting of measured combinatorial variants
- The folder './supervised_regression/' contains a notebook to perform nonlinear, residue-mutation preference regression on various experimental datasets.
  - **Perform supervised logistic regression:** 
  The notebook '01_log_linear_fits_all.ipynb' details how to perform residue-wise logistic fitting of measured combinatorial variant effects.

### Unsupervised scoring and generation of combinatorial variants based on residue-preferences
- The folder './coves/' contains notebooks for unsupervised calculation, sampling and scoring from structural-microenvironment based residue preferences given a PDB file.

  - **Required step: Infer residue preferences from structural information:**
 The notebook '01_infer_residue_preferences_from_structure.ipynb' details how to use the GVP from [Jing et al., 2021](https://github.com/drorlab/gvp) to infer per residue amino acid preferences given the surrounding structural environment. Specifically, this enables inferring the per-residue amino acid preferences from this residues structural surrounding.

  - **Score combinatorial variants using per residue preferences:** 
	The notebook '02_coves_scoring.ipynb' details how to use these inferred preferences to compute combinatorial variant effect scores.

  - **Sample combinatorial variants using per residue preferences:**
	  The notebook '04_sample_coves.ipynb' details how to use the inferred preferences to sample combinatorial amino acid variants.  

  - The remaining notebooks detail analysis for reproducing results in the paper, such as correlating structure-inferred residue preferences with experimentally inferred residue preferences (03_corr_coves_log_reg.ipynb), evaluating CoVES-sampled sequences using the supervised surrogate fitness function (05_coves_sample_eval.ipynb) and for generating plots to estimate generalization error of the surrogate fitness functions (06_TA_generalization_error.ipynb, 07_gfp_roc_90_10.ipynb)

### Additional folders:
- ./src/ contains helper functions for analysis
- ./data/ contains various required data sources of experimental variant effect measurements, PDB structures
- ./other_tools/ contains analysis pipeline for scoring and sampling from ESM-IF, EvCouplings and Protein-MPNN

## Requirements
- Tensorflow 2.0
- Python 3.9
- Torch 1.9.0
- if using, see the respective packages (ESM-IF, EvCoupings, ProteinMPNN) for requirements

Note: Some scripts are tied to our research compute cluster (Harvard O2 research computing with Slurm Workload Manager), and are provided as reference only.

Please feel free to contact (davidding at berkeley dot edu) or (ayshaw at g dot harvard dot edu) for questions.
