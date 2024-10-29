# Anomaly Detection in Small Hydroelectric Plants (SHPs)

This repository contains the code for intelligent fault diagnosis based on the Extended Isolation Forest algorithm, specifically tailored for anomaly detection in Small Hydroelectric Plants (SHPs). This work is based on the methodology presented in the article:

> **de Santis, R. B., & Costa, M. A. (2020).**  
> *Extended isolation forests for fault detection in small hydroelectric plants.*  
> Sustainability, 12(16), 6421.

## Description

The Extended Isolation Forest algorithm is an advancement over traditional isolation forests, aimed at enhancing fault detection accuracy in SHPs by leveraging anomaly detection techniques. This repository includes:
- Preprocessed data files for training and testing,
- Implementation scripts for the isolation forest model, PCA, and KICA-PCA methods, and
- A results file that records the fault detection outputs based on the model’s analysis.

## Requirements

To run the scripts in this repository, ensure that you have the following software and packages installed:

- Python 3.7.6
- SciPy 1.4.1
- Scikit-learn 0.22.1
- Pandas 1.0.1
- NumPy 1.18.1
- Matplotlib 3.1.3

## Citation

If you use this code in your research, please cite the following reference:

```bibtex
@article{de2020extended,
  title={Extended isolation forests for fault detection in small hydroelectric plants},
  author={de Santis, Rodrigo Barbosa and Costa, Marcelo Azevedo},
  journal={Sustainability},
  volume={12},
  number={16},
  pages={6421},
  year={2020},
  publisher={MDPI}
}
```
