Simple repository to play around with different computer vision models and compare their performance on common benchmarks. 

1) Create venv and install requirements:

```python
    python3 -m venv venv
    pip install -r requirements.txt 
```

2) Hyperparam turning and training best models datasets:
```python
    make tune_and_train_all_MNIST
    make tune_and_train_all_CIFAR100
```
3) Visualize results:
```python
    make make_figures
```



