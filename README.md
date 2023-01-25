# A Theory of Link Prediction via Relational Weisfeiler-Leman #

This is the official code base of the paper **A Theory of Link Prediction via Relational Weisfeiler-Leman** based on PyTorch and [TorchDrug], with implemented **Conditional Message Passing Neural Network(C-MPNN)**. It is largely forked from [NBFNet code base](https://github.com/DeepGraphLearning/NBFNet) , with mild modifications to accommodate all models studied in the paper.  Also, it supports training and inference with multiple GPUs or multiple machines. For further detail please refer to the [NBFNet code base](https://github.com/DeepGraphLearning/NBFNet). 

[TorchDrug]: https://github.com/DeepGraphLearning/torchdrug

## Installation ##

You may install the dependencies via either conda or pip. It works with Python 3.7/3.8 and PyTorch version >= 1.8.0.

### From Conda ###

```bash
conda install torchdrug pytorch=1.8.2 cudatoolkit=11.1 -c milagraph -c pytorch-lts -c pyg -c conda-forge
conda install ogb easydict pyyaml -c conda-forge
```

### From Pip ###

```bash
pip install torch==1.8.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install torchdrug
pip install ogb easydict pyyaml
```

## Reproduction ##

To reproduce the experiment in the paper, use the following command. Alternatively, you
may use `--gpus null` to run C-MPNN on a CPU. All the datasets will be automatically
downloaded in the code.

```bash
python script/run.py -c config/inductive/wn18rr.yaml --gpus [0] --version v1
```

We provide the hyperparameters for each experiment in configuration files.
All the configuration files can be found in `config/*/*.yaml`.

For experiments on inductive relation prediction, you need to additionally specify
the split version with `--version v1`.

To run NBFNet with multiple GPUs or multiple machines, use the following commands

```bash
python -m torch.distributed.launch --nproc_per_node=4 script/run.py -c config/inductive/wn18rr.yaml --gpus [0,1,2,3]
```

```bash
python -m torch.distributed.launch --nnodes=4 --nproc_per_node=4 script/run.py -c config/inductive/wn18rr.yaml --gpus [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
```

## Configuration File ##




