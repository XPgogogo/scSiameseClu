# scSiameseClu
**A Siamese Clustering Framework for Interpreting Single-cell RNA Sequencing Data** published in **IJCAI 2025**（CCF-A）.

**Extended Version (arXiv):** [https://arxiv.org/abs/2505.12626](https://arxiv.org/abs/2505.12626)

---

## Overview

Single-cell RNA sequencing (scRNA-seq) enables characterization of cellular heterogeneity, with clustering being key for identifying cell types and marker genes. Despite advances, scRNA-seq data remain challenging due to noise, sparsity, and high dimensionality. Graph neural networks (GNNs) have improved performance but often suffer from over-smoothing, limiting biological interpretability.

**scSiameseClu** is a novel Siamese clustering framework that addresses these issues via three modules:

1. **Dual Augmentation Module**  
   Applies biologically-informed perturbations to the gene expression matrix and cell graph to enhance representation robustness.

2. **Siamese Fusion Module**  
   Combines cross-correlation refinement and adaptive information fusion to capture complex cellular relationships while mitigating over-smoothing.

3. **Optimal Transport Clustering**  
   Utilizes Sinkhorn distance to align cluster assignments with predefined proportions efficiently while maintaining balance.

Comprehensive evaluations on seven real-world datasets show that **scSiameseClu** outperforms state-of-the-art methods in clustering, cell type annotation, and classification.

<img width="848" height="348" alt="scSiameseClu illustration" src="https://github.com/user-attachments/assets/0b142c29-eb92-434e-a1c6-bb9fa0731e3d" />

---

## Conclusion

**scSiameseClu** effectively integrates dual augmentation, siamese fusion, and optimal transport clustering to enhance representation learning while preserving biological relevance.  

Experimental results demonstrate that:

- Outperforms nine baseline methods on scRNA-seq clustering
- Alleviates representation collapse common in GNN-based approaches
- Provides interpretable results for cell type annotation and classification

This framework serves as a powerful tool for advancing single-cell transcriptomics analysis.

---

## Run Example

1. Navigate to pretraining README: `scSiameseClu/scSiameseClu_pretrain/README.md`  
2. Run main training script:  

```bash
python scSiameseClu/main.py
```

Please contact us if you encounter problems during the replication process.


## Requirements
We implement scSGC in Python 3.7 based on PyTorch (version 1.12+cu113).
All required dependencies have been carefully summarized and are now included in the requirements.txt file. Users can simply run the following command to install all necessary packages: pip install -r requirements.txt

```shell
Keras==2.4.3
numpy～=1.21.6
pandas==1.3.5
scanpy～=1.9.3
torch==1.12.0
h5py==3.8.0
loguru==0.7.3
scikit-learn==0.22.1
torchmetrics==0.10.3
matplotlib==3.5.3
umap-learn==0.5.3
importlib-metadata==1.7.0
```

Please note that if using different versions, the results reported in our paper might not be able to repeat.

## The raw data
Setting data_file to the destination to the data (stored in h5 format, with two components X and Y, where X is the cell by gene count matrix and Y is the true labels), n_clusters to the number of clusters.

In order to ensure the accuracy of the experimental results, we conducted more than 10 times runs on all the datasets and reported the mean and variance of these running results, reducing the result bias caused by randomness and variability, so as to obtain more reliable and stable results. Hyperparameter settings for all datasets can be found in the code.

[The raw data used in this paper can be found: https://github.com/XPgogogo/scSGC/tree/master/datasets](https://github.com/XPgogogo/scCDCG/tree/master/datasets)
<img width="1076" height="420" alt="image" src="https://github.com/user-attachments/assets/6e797adc-f886-466c-9e1e-9fbe21c40bbf" />



## 📚 Citation
Please cite our paper if you use this code or or the dataset we provide in your own work:

```
@article{xu2025scsiameseclu,
  title={scsiameseclu: A siamese clustering framework for interpreting single-cell rna sequencing data},
  author={Xu, Ping and Ning, Zhiyuan and Li, Pengjiang and Liu, Wenhao and Wang, Pengyang and Cui, Jiaxu and Zhou, Yuanchun and Wang, Pengfei},
  journal={arXiv preprint arXiv:2505.12626},
  year={2025}
}
```

# Contact
Ph.D student Ping XU

Computer Network Information Center, Chinese Academy of Sciences

University of Chinese Academy of Sciences

No.2 Dongshen South St

Beijing, P.R China, 100190

Personal Email: xuping0098@gmail.com

Official Email: xuping@cnic.cn
