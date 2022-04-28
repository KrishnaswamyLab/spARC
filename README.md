# spARC: Spatial Affinity-Graph Recovery of Counts
Data diffusion based methods for denoising and integrating spatial transcriptomics data.

Spatial transcriptomics data allows for analysis to be performed between spatially distinct populations of cells by coupling in situ location with genes expression profiles. While powerful, this additional information is accompanied by a high degree of technical noise and artifact, leading to incomplete RNA profiling of cells and difficulties with downstream analysis. Most imputation techniques that attempt to recover expression dynamics only take into account the gene expression of a single cell and not the cell's spatial location. A cell's spatial location, however, has a great deal of information about expression properties as spatial neighborhoods provide strong evidence for co-occuring molecular phenotypes (Palla et al., 2022). With these insights, we developed spARC, a diffusion geometric framework that integrates in situ location and gene expression information to denoise spatial transcriptomics data and identify paracrine receptor-ligand signaling interactions between cells within their spatial contexts.

# Overview of Algorithm:

![alt text](https://github.com/KrishnaswamyLab/spARC/blob/main/Figures/Algorithm%20Overview.png)

spARC recovers ground truth gene expression values by creating an integrated diffusion operator that takes into account both spatial location and gene expression profiles of every cell. By first computing diffusion operators based on spatial location and gene expression profiles, we can create an integrated operator that we can apply directly to input gene expression values to recover ground truth counts.



# Getting Started

To install please use:

`pip install git+https://github.com/KrishnaswamyLab/spARC`

For overview of functionality, please take a look at our [spARC tutorial](https://github.com/KrishnaswamyLab/spARC/blob/main/tutorial/SPARC-tutorial.ipynb).



