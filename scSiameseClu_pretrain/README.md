# Pre-training For scSiameseClu

## Quick Start

- Step1: Pre-train Auto-Encoder (AE)

  ```
  python ./pretrain_ae/main.py
  ```

- Step2: Pre-train Improved Graph Auto-Encoder ([IGAE](https://arxiv.org/abs/2012.09600))

  ```
  python ./pretrain_gae/main.py
  ```

- Step3: Joint Pre-training

  ```
  python ./pretrain/main.py
  ```
  
  

