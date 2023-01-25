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
For experiments on inductive relation prediction, you need to additionally specify
the split version with `--version v1`.

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


`config/inductive/CMPNN-test/*.yaml`
stores all the config files needed for reproducing **Inductive Relation Prediction Experiments**, with different model instances labelled in the files. The naming and the corresponding model variation are shown below

|                    | Model Choice                                                                    | Prefix in .yaml |
|--------------------|---------------------------------------------------------------------------------|-----------------|
| **Aggregate Function** | Principal Neighborhood Aggregation(PNA)                                                                             | `-pna `           |
|                    | sum                                                                             | `-sum  `          |
| **Message Function**   |  $\theta_r^{1}(\mathbf{h}_{w \mid u,q}^{(t)},\mathbf{z}_q) =  \mathbf{h}_{w \mid u,q}^{(t)} * \mathbf{W}_{r}^{(t)} \mathbf{z}_q $ | `-dep `           |
|                    | $ \theta_r^{2}(\mathbf{h}_{w \mid u,q}^{(t)},\mathbf{z}_q) = \mathbf{h}_{w \mid u,q}^{(t)} * \mathbf{b}_r $              | `-indep  `        |
|                    | $ \theta_r^{3}(\mathbf{h}_{w \mid u,q}^{(t)},\mathbf{z}_q) = \mathbf{W}_{r}^{(t)}\mathbf{h}_{w \mid u,q}^{(t)} $         |` -rgcn `          |
|                    | $ \theta_r^{4}(\mathbf{h}_{w \mid u,q}^{(t)},\mathbf{z}_q) = \mathbf{W}_{r}^{(t)}\mathbf{h}_{w \mid u,q}^{(t)} * \mathbf{z}_q $ | `-rgcn_query`     |
| **History Function**   | $f(t) = t$                                                                       | `-t`              |
|                    | $f(t) = 0$                                                                        | `-0`              |


### Initialization ###

`config/inductive/Initialisation_test/*.yaml`
stores all the config files for reproducing **Initialization Experiments**, with different initialization methods. Note that the config file of *Query* are shown in previous *Inductive Relation Prediction Experiment*.

The naming and the corresponding model variation are shown below.

| Initialization | Formula                                                                   | Prefix in .yaml |
|----------------|---------------------------------------------------------------------------|-----------------|
| AllZero        | $\delta_0(u,v,q) = \mathbf{0}$                                            | `-0  `            |
| Zero-One       | $\delta_1(u,v,q) = \mathbb{1}_{u = v} * \mathbf{1}$                       | `-0-1 `           |
| Query          | $\delta_2(u,v,q) = \mathbb{1}_{u = v} * \mathbf{z}_q $                           |                 |
| QueryWithNoise | $\delta_3(u,v,q)  = \mathbb{1}_{u = v} * (\mathbf{z}_q + \mathbf{\epsilon}_{u})$ | `-QueryWithNoise` |
| AllNoiseQuery  | $\delta_4(u,v,q) = (\mathbb{1}_{u = v} *\mathbf{z}_q) + \mathbf{\epsilon}_{u}$   | `-AllNoiseQuery`  |
| RandomQuery    | $\delta_5(u,v,q) = \mathbb{1}_{u = v} * \mathbf{\epsilon}_{q} $           | `-rand-query `    |



