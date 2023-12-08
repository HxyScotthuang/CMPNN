# A Theory of Link Prediction via Relational Weisfeiler-Leman on Knowledge Graphs #

This is the official code base of the NeurIPS 2023 paper **A Theory of Link Prediction via Relational Weisfeiler-Leman on Knowledge Graphs** [(ArXiv)](https://arxiv.org/abs/2302.02209) based on PyTorch and [TorchDrug], with implemented **Conditional Message Passing Neural Network (C-MPNN)**. It is largely based on the [NBFNet code base](https://github.com/DeepGraphLearning/NBFNet) , with mild modifications to accommodate all models studied in the paper.  Also, it supports training and inference with multiple GPUs or multiple machines. 

[TorchDrug]: https://github.com/DeepGraphLearning/torchdrug

### Installation ###

```bash
pip install torch
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
For experiments on inductive relation prediction, you need to additionally specify
the split version with `--version v1`.

For CPU only, run the following command
```bash
python script/run.py -c config/inductive/wn18rr.yaml --gpus null --version v1
```

To run C-MPNN with multiple GPUs or multiple machines, use the following commands

```bash
python -m torch.distributed.launch --nproc_per_node=4 script/run.py -c config/inductive/wn18rr.yaml --gpus [0,1,2,3]
```

```bash
python -m torch.distributed.launch --nnodes=4 --nproc_per_node=4 script/run.py -c config/inductive/wn18rr.yaml --gpus [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
```

## Configuration ##
We provide the hyperparameters for each experiment in configuration files.
All the configuration files can be found in `config/*/*.yaml`.

### Inductive Relation Prediction Experiments ###
The naming and the corresponding model variation are shown below. 

|                    | Model Architecture Choice                                                                    |  Key      | Value | 
|--------------------|---------------------------------------------------------------------------------|-------|-------|
| **Aggregate Function** | Principal Neighborhood Aggregation(PNA)                                                                             | `aggregate_func`| `pna`|
|                    | Sum                                                                             |  | `sum` |
| **Message Function**   |  ${Mes}_r^{1}(\mathbf{h}\_{w \mid u,q}^{(t)},\mathbf{z}\_q) =  \mathbf{h}\_{w \mid u,q}^{(t)} * \mathbf{W}\_{r}^{(t)} \mathbf{z}\_q $ |  (`dependent`, `rgcn`)   | `(yes,no)` |
|                    | ${Mes}_r^{2}(\mathbf{h}\_{w \mid u,q}^{(t)},\mathbf{z}\_q) = \mathbf{h}\_{w \mid u,q}^{(t)} * \mathbf{b}\_r $              | |`(no,no)`|
|                    | ${Mes}_r^{3}(\mathbf{h}\_{w \mid u,q}^{(t)},\mathbf{z}\_q) = \mathbf{W}\_{r}^{(t)}\mathbf{h}\_{w \mid u,q}^{(t)} $         |  |`(_,yes)`|                 
| **History Function**   | $f(t) = t$                                                                       | `set_boundary` | `no`|
|                    | $f(t) = 0$                                                                        |  | `yes`|

In addition, if we consider using ${Mes}_r^3$, then we can pass in additional parameter `num_bases: k` where k is the number of basis for basis decomposition. 


### Initialization Experiments ###

The naming in the config file and the corresponding model variation are shown below.

| Initialization  | Equation                                                                   | 
|----------------|---------------------------------------------------------------------------|
| AllZero        | ${Init}_0(u,v,q) = \mathbf{0}$                                            | 
| Zero-One       | ${Init}_1(u,v,q) = \mathbb{1}\_{u = v} * \mathbf{1}$                       | 
| Query          | ${Init}_2(u,v,q) = \mathbb{1}\_{u = v} * \mathbf{z}\_q $                           |               
| QueryWithNoise | ${Init}_3(u,v,q)  = \mathbb{1}\_{u = v} * (\mathbf{z}\_q + \mathbf{\epsilon}\_{u})$ | 

### Transductive Experiments ###

For experiments on transductive relation prediction:
```bash
python script/run.py -c config/knowledge_graph/wn18rr.yaml --gpus [0] 
```

### Readout Experiments ###

The **TRI-SQR** dataset and synthetic experiments are shown in `TRI-SQR dataset.ipynb`

The key and acceptable values in the config file:
| Key |  Value |
|----------------| -----------------|
| `has_readout`        | `yes` / `no`   | 
| `readout_type`     | `sum`/ `mean`                   | 
| `query_specific_readout`         | `yes` / `no`                          |    

For further details please refer to the [NBFNet code base](https://github.com/DeepGraphLearning/NBFNet). 

```latex
@inproceedings{
huang2023a,
title={A Theory of Link Prediction via Relational Weisfeiler-Leman on Knowledge Graphs},
author={Xingyue Huang and Miguel Romero Orth and İsmail İlkan Ceylan and Pablo Barceló},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=7hLlZNrkt5}
}

```
