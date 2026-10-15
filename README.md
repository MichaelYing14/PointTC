# PointTC

Code for **Joint Semantic-Channel Coding and Modulation for Token Communications**, IEEE TWC, 2026. [Paper](https://doi.org/10.1109/TWC.2025.3636174).

Supports PointTC, PointTC+CA and PointTC+CA+RA over AWGN and Rayleigh channels, with 2,048 points, 64-QAM and symbol budgets of 50, 100, 150, 200, 250 or 300. The six method/channel configurations default to 300 symbols. Only the **PointTC+CA+RA AWGN, maximum 300 symbols** pretrained weight is provided. Configuration IDs `pointtc`, `ca` and `ca_ra` refer to these three methods, respectively.

```text
pointtc/          Models, data loading and metrics
configs/          Six method/channel configurations
utils/pc_error    Point-cloud evaluation executable
train.py          Training and finetuning
evaluate.py       Dataset evaluation
infer.py          Single point-cloud reconstruction
```

**Weight download:** [pointtc_ca_ra_awgn.pt (Google Drive)](https://drive.google.com/file/d/1Aq7OyVp1dJqqNQTw321flv2p8pxErdg5/view?usp=sharing). Save it as `pointtc_ca_ra_awgn.pt` in the repository root.

## Installation

The code was tested with Python 3.9, PyTorch 2.0.1/CUDA 11.7 and PyTorch3D 0.7.7.

```bash
conda create -n pointtc python=3.9 pip
conda activate pointtc
conda install pytorch==2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia
python -m pip install -r requirements.txt
python -m pip install iopath
python -m pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7'
```

Building PyTorch3D requires a compiler and, for CUDA, a compatible toolkit  (see its [installation guide](https://github.com/facebookresearch/pytorch3d/blob/v0.7.7/INSTALL.md)).

## Data

Follow the data preparation referenced by [SEPT](https://github.com/aprilbian/SEPT), using the prepared **ShapeNetCore.v2** dataset from [PointCloudDatasets](https://github.com/AnTao97/PointCloudDatasets#download-link). It contains 35,708 training, 5,158 validation and 10,261 test objects.

Place `train*.h5`, `val*.h5` and `test*.h5` together.  Use the prepared normalized coordinates unchanged; the loader performs no additional sampling or normalization.

Replace `/path/to/shapenetcorev2_hdf5_2048` in the commands below with your dataset directory.

## Evaluation

```bash
python evaluate.py --config configs/ca_ra_awgn.json \
  --checkpoint pointtc_ca_ra_awgn.pt --data-root /path/to/shapenetcorev2_hdf5_2048 \
  --output-dir outputs/eval --device cuda:0
```

This evaluates **Chamfer distance, D1 PSNR, D2 PSNR and transmitted symbols** at 0:2.5:15 dB, prints the results and saves per-object and summary CSVs. It automatically uses the included `utils/pc_error` (version 0.13.4, Linux x86-64) and no extra metric argument is needed. On other platforms, use a compatible [pc_error](https://github.com/minhkstn/mpeg-pcc-dmetric) executable with `--pc-error`.

## Training

PointTC can be trained at a chosen fixed SNR. The provided configurations use 10 dB by default. PointTC+CA and PointTC+CA+RA sample training SNRs from a range, set to [-0.5, 10.5] dB in the provided configurations. Change `train_snr` in the configuration file to select a different fixed SNR or range.

Set `num_symbols` in the configuration file to select the total symbol count for PointTC and PointTC+CA, or the maximum for PointTC+CA+RA. The main/auxiliary split is selected automatically for all three methods and both channels:

| `num_symbols` | Main branch | Auxiliary branch |
| ---: | ---: | ---: |
| 300 | 200 | 100 |
| 250 | 200 | 50 |
| 200 | 160 | 40 |
| 150 | 120 | 30 |
| 100 | 80 | 20 |
| 50 | 40 | 10 |

For example, `"num_symbols": 150` selects 120 main and 30 auxiliary symbols. PointTC+CA+RA sends the main branch plus 1/5, 2/5, 3/5, 4/5 or all of the auxiliary branch. Use a separate output directory for each budget, and use the matching configuration for evaluation and inference.

Train PointTC and PointTC+CA independently, initialize PointTC+CA+RA from PointTC+CA, and initialize each Rayleigh model from its AWGN counterpart.

```bash
python train.py --config configs/pointtc_awgn.json --data-root /path/to/shapenetcorev2_hdf5_2048 \
  --output-dir outputs/pointtc_awgn --device cuda:0
python train.py --config configs/ca_awgn.json --data-root /path/to/shapenetcorev2_hdf5_2048 \
  --output-dir outputs/ca_awgn --device cuda:0
python train.py --config configs/ca_ra_awgn.json --data-root /path/to/shapenetcorev2_hdf5_2048 \
  --pretrained outputs/ca_awgn/best.pt --output-dir outputs/ca_ra_awgn --device cuda:0
python train.py --config configs/ca_ra_rayleigh.json --data-root /path/to/shapenetcorev2_hdf5_2048 \
  --pretrained outputs/ca_ra_awgn/best.pt --output-dir outputs/ca_ra_rayleigh --device cuda:0
```

Use `--resume outputs/RUN/last.pt` to restore a training checkpoint, or `--pretrained` to initialize weights. The checkpoint and target model must have the same `num_symbols`, including PointTC+CA to PointTC+CA+RA initialization. The public weight supports initialization at 300 symbols, not optimizer-state resumption. 

## Single-cloud inference

Supply normalized `(2048, 3)` XYZ points in `.npy` or `.ply` format.

```bash
python infer.py --config configs/ca_ra_awgn.json --checkpoint pointtc_ca_ra_awgn.pt \
  --input example.npy --output outputs/reconstruction.ply --snr 10 --device cuda:0
```

## Citation

```bibtex
@article{ying2026pointtc,
  author={Ying, Jingkai and Qin, Zhijin and Feng, Yulong and Wang, Liejun and Tao, Xiaoming},
  journal={IEEE Transactions on Wireless Communications}, 
  title={Joint Semantic-Channel Coding and Modulation for Token Communications}, 
  year={2026},
  volume={25},
  number={},
  pages={8179-8193},
  keywords={Point cloud compression;Transformers;Semantic communication;Artificial intelligence;Encoding;Communication systems;Feature extraction;Adaptation models;Symbols;Image reconstruction;Token communication;joint semantic-channel coding;digital modulation;adaptive transmission;point cloud},
  doi={10.1109/TWC.2025.3636174}}
```

## Acknowledgments

Project code is released under [MIT](LICENSE). We acknowledge [SEPT](https://github.com/aprilbian/SEPT) for code and module design references, [PointCloudDatasets](https://github.com/AnTao97/PointCloudDatasets) for the data workflow, and [PyTorch3D](https://github.com/facebookresearch/pytorch3d) and [MPEG pc_error](https://github.com/minhkstn/mpeg-pcc-dmetric) for the tools. The bundled pc_error has its own [license notices](utils/LICENSE.pc_error). Other dependencies and datasets retain their respective licenses.

We also thank OpenAI Codex for assistance in organizing and preparing the code for this open-source release.
