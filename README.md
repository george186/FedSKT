# FedSKT

Implementation of the paper *"FedSKT: A Selective Knowledge Transfer Scheme for Federated Learning via End-Edge-Cloud Collaboration"*.

## Preparation

Requirements: Python 3.9, PyTorch 2.2.2 (CUDA 12.1).

```bash
pip install -r requirements.txt
```

Place the CIFAR-10 dataset under `./data`, i.e. `./data/cifar-10-batches-py/`:

```bash
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./data', train=True, download=True)"
```

## Run

```bash
python main_fedskt.py \
    --dataset cifar10 --data_dir ./data \
    --client_number 225 --edge_number 15 \
    --comm_round 1000 --batch_size 16 --lr 0.001 \
    --r 3 --s 2 --T_agg 3
```

The test accuracy of the cloud model is printed after each communication round.

## Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--data_dir` | `./data` | Directory containing `cifar-10-batches-py`. |
| `--client_number` | `225` | Total number of clients. |
| `--edge_number` | `15` | Number of edges; must divide `client_number`. |
| `--comm_round` | `1000` | Number of communication rounds. |
| `--batch_size` | `16` | Batch size. |
| `--lr` | `0.001` | Local learning rate. |
| `--r` | `3` | Knowledge-distillation rounds between nodes in each round. |
| `--s` | `2` | Data-selection ratio: `1/s` of the transmitted samples are used. |
| `--T_agg` | `3` | Temperature of the knowledge-transfer loss. |

## Reference

The implementation of this work refers to [FedAgg](https://github.com/wuzhiyuan2000/FedAgg): the end-edge-cloud node hierarchy, the autoencoder, and part of the training code are referenced and adopted from FedAgg, while FedSKT introduces selective knowledge transfer (teacher selection and data selection).
