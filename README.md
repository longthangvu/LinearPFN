# LinearPFN

## Reproduce TSLib Results

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Download the Time-Series-Library datasets. THUML documents the datasets in the [Time-Series-Library repo](https://github.com/thuml/Time-Series-Library) and hosts a mirror at [thuml/Time-Series-Library](https://huggingface.co/datasets/thuml/Time-Series-Library). The evaluation scripts expect the dataset root to contain folders such as `ETT-small`, `electricity`, `exchange_rate`, `traffic`, and `weather`.

One way to install them is:

```bash
git lfs install
git clone https://huggingface.co/datasets/thuml/Time-Series-Library ../Time-Series-Library/dataset
```

Run the reproduced evaluation with the public [LinearPFN weights](https://huggingface.co/longthangvu/LinearPFN):

```bash
bash scripts/test_tslib.sh --load_from hf 
```

| Dataset | MSE | MAE |
| --- | ---: | ---: |
| electricity | 0.30975318 | 0.38683423 |
| ETTh1 | 0.0679849 | 0.1959967 |
| ETTh2 | 0.12804282 | 0.27522495 |
| ETTm1 | 0.03448291 | 0.13719921 |
| ETTm2 | 0.08038426 | 0.20534784 |
| exchange | 0.12550306 | 0.26700076 |
| traffic | 0.24860084 | 0.34372288 |
| weather | 0.001456768 | 0.030810302 |
