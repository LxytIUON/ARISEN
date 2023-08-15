# ARISEN

This is the official implementation of the paper "**Sequential recommendation via an adaptive cross-domain knowledge decomposition**" based on PyTorch.

This paper was accepted by CIKM2023.

## Overview

Included here are model codes for the ARISEN model and several sequence baselines implemented on its basis, including DIN, IV4Rec, GRU4Rec, and SSE-PT.

## Experimental Setting

All hyperparameter settings for the ARISEN and ARISEN + baselines on the AMAZON dataset can be found in the `config` folder. The data settings can be found in the file `config/config.py`.

## Dataset

Since the commercial datasets, i.e., lenovo, are proprietary industrial datasets, here we release the experimental settings and data-processing codes of the AMAZON dataset.

## Train and evaluate models:

Please run the `main.py` for model train and test.

### Environments

The following python packages are required:

```
python==3.9
torch==1.13.0
numpy==1.21.5
pandas==1.4.4
scikit-learn==1.0.2
tqdm==4.64.1
```



The experiments are based on the following environments:

- CUDA Version: 11.7
- OS: Ubuntu 18.04
- GPU: NVIDIA Tesla V100 GPU
- CPU: Intel(R) Xeon(R) Gold 5118 CPU (12 cores, 2.30GHz)

## Connect

If you have any questions, feel free to contact us through email lxyt7642@tju.edu.cn or GitHub issues. Thanks!
