# FedSKT: A Selective Knowledge Transfer Scheme for Federated Learning via End-Edge-Cloud Collaboration

Official PyTorch implementation of the paper **"FedSKT: A Selective Knowledge Transfer Scheme for Federated Learning via End-Edge-Cloud Collaboration"**.

FedSKT studies federated learning on a three-tier **cloud&ndash;edge&ndash;client** hierarchy, where a large cloud model and a set of lightweight edge models collaborate with a large number of data-isolated clients. Instead of exchanging model parameters or raw samples, knowledge is transferred through the **latent space of a pre-trained autoencoder**, and the transfer is made *selective* in two complementary ways:

1. **Selective teacher assignment** &mdash; for every adjacent pair of nodes the teacher/student direction is decided online by comparing the two models' accuracy on the shared latent representations, so that knowledge always flows from the better model to the worse one.
2. **Selective data transfer** &mdash; only a fraction `1/s` (`s` is a hyper-parameter) of the shared latent representations is sampled in each local iteration, which cuts communication and computation while keeping the useful signal.

Because only autoencoder latents travel across nodes (and are decoded locally by the receiver's own decoder), the scheme keeps raw training data on the devices.

---

## Architecture

```
                      +---------------------+
                      |   Cloud  (ResNet-18) |
                      +----------+----------+
                                 |  latent representations
                 +---------------+---------------+
                 |                               |
         +-------+-------+               +-------+-------+
         | Edge 0 (R10)  |      ...      | Edge 14 (R10) |
         +-------+-------+               +-------+-------+
                 |                               |
     +-----------+-----------+       +-----------+-----------+
     |           |           |       |           |           |
   +-+-+       +-+-+       +-+-+   +-+-+       +-+-+       +-+-+
   |C0 |  ...  |C14|       |C15|   |C |  ...  |C |       |C  |   <- CNN clients
   +---+       +---+       +---+   +---+       +---+       +---+
```

* `--client_number` clients are grouped into `--edge_number` edges (`client_number % edge_number == 0`).
* Every node holds a copy of the shared pre-trained autoencoder (`params.pkl`) that maps an image to a compact latent code and back.
* Each communication round performs a bottom-up traversal: leaf clients first exchange knowledge with their edge, then edges with the cloud.

---

## Repository structure

| File | Description |
| --- | --- |
| `main_fedagg.py` | Entry point: argument parsing, data loading, cloud / edge / client model construction, training launch. |
| `fedskt.py` | Core of FedSKT: the node tree (`Node`), initialisation, recursive `train_FedSkt`, bi-directional selective knowledge transfer (`BSBODP` / `BSBODP_TS`), selective teacher selection (`update_teacher`), data selection (`data_selection`) and the composite losses. |
| `autoencoder_pretrained.py` | Autoencoder definition (3&ndash;16&ndash;8&ndash;4 encoder, mirrored decoder) and the loader `create_autoencoder()`. |
| `params.pkl` | Pre-trained weights of the autoencoder used by FedSKT. |
| `model_zoo.py` | Model factory: `resnet18` (cloud), `resnet10` (edge), `cnn` (client). |
| `data_loader.py` | CIFAR-10 loading and non-IID partitioning into `client_number` clients. |
| `datasets.py` | Truncated CIFAR-10 / CIFAR-100 dataset wrappers. |
| `utils.py` | Utilities: KL / CE losses, top-k accuracy, running average, parameter flattening (adapted from [FedML](https://github.com/FedML-AI/FedML)). |
| `requirements.txt` | Python dependencies. |

---

## Environment

* Python 3.9
* PyTorch 2.2.2 + CUDA 12.1, torchvision 0.17.2
* A single CUDA GPU is required (the code calls `.cuda()` unconditionally)

```bash
pip install -r requirements.txt
```

---

## Dataset preparation

CIFAR-10 is expected in `./data` in the raw PyTorch layout (`./data/cifar-10-batches-py/`); `data_loader.py` builds the truncated dataset with `download=False`, so please fetch it beforehand:

```bash
python -c "from torchvision.datasets import CIFAR10; CIFAR10('./data', train=True, download=True)"
```

Alternatively, run `main_fedagg.py` with `--data_dir` pointing to the directory that already contains `cifar-10-batches-py`.

---

## Training

```bash
python main_fedagg.py \
    --dataset cifar10 \
    --data_dir ./data \
    --client_number 225 \
    --edge_number 15 \
    --comm_round 1000 \
    --batch_size 16 \
    --lr 0.001 \
    --r 3 \
    --s 2 \
    --T_agg 3
```

The cloud model's accuracy on the global test set is printed after every communication round:

```
Test/AccTop1 in comm_round <round> <accuracy>
```

### Main arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--dataset` | `cifar10` | Dataset name. |
| `--data_dir` | `./data` | Directory holding `cifar-10-batches-py`. |
| `--client_number` | `225` | Total number of clients. |
| `--edge_number` | `15` | Number of edges; must divide `client_number`. |
| `--comm_round` | `1000` | Number of communication rounds. |
| `--batch_size` | `16` | Batch size. |
| `--lr` | `0.001` | Local learning rate (SGD, momentum 0.9, weight decay 1e-3). |
| `--r` | `3` | Local optimisation iterations per node and round. |
| `--s` | `2` | Selective data-transfer ratio: `1/s` of the shared latents is used. |
| `--T_agg` | `3` | Distillation temperature (and the KL weight `alpha`) of the transfer loss. |
| `--partition_method` | `hetero` | Non-IID partition strategy. |
| `--partition_alpha` | `3.0` | Dirichlet concentration parameter of the partition. |

---

## Citation

If you find this code useful, please cite the paper:

```bibtex
@article{fedskt,
  title   = {FedSKT: A Selective Knowledge Transfer Scheme for Federated Learning
             via End-Edge-Cloud Collaboration},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

---

## Acknowledgements

Parts of `utils.py`, `datasets.py` and `data_loader.py` are adapted from [FedML](https://github.com/FedML-AI/FedML).

## Contact

Questions and suggestions are welcome via GitHub issues.
