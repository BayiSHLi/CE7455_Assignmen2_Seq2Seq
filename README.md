# CE7455-Assignment-2: Seq2Seq model for machine translation

This is the repositoriy for the assignment of CE7455 course at CCDS, Nanyang Technological University. The task is to implement a Seq2Seq model for machine translation using PyTorch.

## Implementation Environment
- Python 3.12
- PyTorch 2.5.1
- cuda 12.1
- torchtext 0.15.2

## Code base
Please find the code base at Final_ipynb.ipynb

## Models
Please find the modified models for tasks of the assignment at '/models' directory

## Results

|          | Rouge 1 | - | - | Rouge 2   | - | - |
|   ----    |----|----|----|-----------|----|----|
| Model | F-measure | Precision | Recall | F-measure | Precision | Recall |
| GRU | 0.45 | 0.46 | 0.45 | 0.25      | 0.26 | 0.25 |
| LSTM | 0.46 | 0.47 | 0.46 | 0.26      | 0.27 | 0.26 |
| bi-LSTM | 0.47 | 0.48 | 0.47 | 0.27      | 0.28 | 0.27 |
| Attention | 0.48 | 0.49 | 0.48 | 0.28      | 0.29 | 0.28 |
| Transformer | 0.49 | 0.50 | 0.49 | 0.29      | 0.30 | 0.29 |

## Evaluate the results
To evaluate the results, please download the related checkpoint file from the link below and place it in the root directory of the project.

To eval the results, please use the command below:
```python test.py --task <GRU/LSTM/bi-LSTM/Attention/Transformer> --eval --ckpt <checkpoint_path>```

## Reproduction of Training
To reproduce the results, please use the commands below:
```python test.py --task <GRU/LSTM/bi-LSTM/Attention/Transformer> --n_epochs 20```



