# Classical Matrix Factorization Algorithm

This project is the implement of classical matrix factorization algorithm for Collaborative Filtering including:

- SVD
- SVD with Bias
- PMF
- NMF

# Requirements

We run the code on python 3.6.8, so you'd better use the python version not lower than 3.6.8.

We also use the Anaconda as the environment manager.

You can install the requirements by:

```bash
conda env create -f environment.yml
pip install -r requirement.txt
```

# Train model

## PMF

You can run the PMF algorithm by:

```bash
python ./lib/classic_algo/pmf.py
```

you can change the comment in main() to change the config, then you can obtain the results.

## SVD, SVD_Bias, NMF

The config for these three algorithms is in cfg/config.py. The most important config is in config.exp.

```python
config.exp.feature_num_list		# the number of features
config.exp.algo_list			# the algorithms, 'svd', 'svd_bias', 'nmf'
config.exp.gamma_list			# the learning rate
config.exp.lamb_list			# the regularization factor
```

Then you can simply run

```bash
python main.py
```

The final output result is in outputs.