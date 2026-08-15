# Rigid 3D Object Alignment: Optimization vs. Feed-Forward Prediction

Code for *Rigid 3D Object Alignment: Optimization vs. Feed-Forward Prediction*.

The task: given an independently generated 3D asset (**src**) and a base object (**tgt**),
predict the rigid transformation that places the asset onto the base — without modifying
either geometry. The transformation is defined relative to the asset's centroid,

$$\Phi(\mathbf{x};\tau) = \mathbf{R}(\mathbf{x} - \mathbf{c}_{src}) + \mathbf{c}_{src} + \mathbf{t}$$

so the learned rotation is decoupled from the global displacement.

This repository contains the **feed-forward Vision–Language–Action (VLA)** predictor, which
regresses $\tau$ in a single sub-second forward pass.

![VLA Pipeline Architecture](https://github.com/user-attachments/assets/a0d9437a-d4e1-425a-8f01-0e8e7141a3f4)
## Two pipelines

Both are the same model and the same training script; they differ only in whether the prompt
carries reference views of the correctly assembled scene.

| Config | Reference views | Inputs |
|---|---|---|
| `assets_aligner` | no | 3 canonical views of the unaligned pair (ZY, ZX, XY) |
| `assets_aligner_ref` | yes | the same 3 views, plus 6 renderings of the assembled scene |

Reference-free is the headline setting: it needs nothing but the asset and base at inference
time. The reference-view variant is the ablation that receives renderings of the correct
answer's *appearance* as extra conditioning.


## Installation

Requires Python 3.11+, an NVIDIA GPU, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/MacPaw/rigid-3d-alignment.git && cd rigid-3d-alignment && uv sync
```

Minimum VRAM: >10 GB for inference, >70 GB for full fine-tuning.

## Datasets

Two Hugging Face datasets. Everything needed for training and prediction is fetched
automatically; only the mesh archive used for scoring is a manual step.

**[`macpaw-research/asset-alignment-pairs-905k`](https://huggingface.co/datasets/macpaw-research/asset-alignment-pairs-905k)**
is the main dataset: ~905k asset–base pairs with ground-truth transformations, canonical
multi-view renderings, depth, and Set-of-Marks visual prompts.

About half of those pairs carry a rotation as well as a translation; the rest are
translation-only. The pipeline trains and evaluates on the rotation-bearing half (~452k
pairs).The dataset is *streamed*, so it is never materialized on disk.

The same repo also carries **`meshes.tar`**, the canonical source meshes as
`<asset_id>/part_<part_id>.glb`. It is **~1.2 TB extracted** and is *not* downloaded
automatically. Training and prediction never touch it — it is required only by
`evaluate.py`, whose ADD-S, Chamfer, and normalized-translation metrics are computed against
real geometry. See [Evaluate](#evaluate) for how to extract just the parts you need.

**[`macpaw-research/asset-alignment-reference-views`](https://huggingface.co/datasets/macpaw-research/asset-alignment-reference-views)**
holds the reference views — renderings of each correctly assembled pair from two planes
(XY, ZY) at three angles (30°, 90°, 270°) — plus `metadata/index_map.csv`, which maps
`(asset_name, part_idx)` to a row in that dataset. It is only used by `assets_aligner_ref`.

## Usage
Before running the pipeline, you need to manually download the normalization statistics for the assets:

```bash
uv run hf download macpaw-research/asset-alignment-pairs-905k --include "assets/*" --repo-type dataset --local-dir .
```

Every entrypoint takes the config name as its first argument.

### Train

```bash
uv run python scripts/train.py assets_aligner --exp-name my_run
```

Swap in `assets_aligner_ref` to train with reference views. Add `--resume` to continue from
the latest checkpoint. Checkpoints are written to
`checkpoints/<config-name>/<exp-name>/<step>/`. Training is single-GPU.

### Predict

`predict.py` runs a checkpoint over one split and writes per-sample ground truth and
predictions, in real units, for offline scoring.

```bash
uv run python scripts/predict.py assets_aligner --exp-name eval --out predictions_test.json
```

With no weights given, it downloads the released checkpoint for that config — 
[`qwen3-dual-dit-aligner`](https://huggingface.co/macpaw-research/qwen3-dual-dit-aligner) for
`assets_aligner`, [`qwen3-dual-dit-aligner-ref`](https://huggingface.co/macpaw-research/qwen3-dual-dit-aligner-ref)
for `assets_aligner_ref`. To score your own run instead, point at its directory, which must
contain `model.safetensors`:

```bash
uv run python scripts/predict.py assets_aligner --exp-name eval --pytorch-weight-path checkpoints/assets_aligner/my_run/10608 --out predictions_test.json
```

Options: `--split` (default `test`), `--max-samples` (required with `--split train`, which
shuffles and iterates indefinitely), and `--out`.

### Evaluate

`evaluate.py` scores a prediction dump against the canonical source meshes. It does not load
the model.

```bash
uv run python scripts/evaluate.py --predictions predictions_test.json --mesh_dir /path/to/meshes --output_json eval_results.json
```

`--mesh_dir` must contain `<asset_id>/part_<part_id>.glb`. Reports translation RMSE and L2,
normalized translation error, geodesic rotation error in degrees, ADD-S with accuracy at 10%
of object diameter, and Chamfer distance.
#### Getting the meshes

The meshes live in `meshes.tar` in the main dataset repo and come to **~1.9 TB extracted**

### Recomputing normalization statistics

Only needed for a new dataset.

```bash
uv run python scripts/compute_norm_stats.py --config-name=assets_aligner --max-frames=20000
```


## Credits

This work stands on several open releases. Please cite them alongside ours where relevant.

**[openpi](https://github.com/Physical-Intelligence/openpi)** — Physical Intelligence. The
data transform pipeline, normalization utilities, training configuration system, and array
typing helpers in this repository are derived from it.

```bibtex
@misc{openpi,
  title        = {openpi},
  author       = {{Physical Intelligence}},
  howpublished = {\url{https://github.com/Physical-Intelligence/openpi}}
}
```

**[Xiaomi-Robotics-0](https://github.com/XiaomiRobotics/Xiaomi-Robotics-0)** — the VLA
backbone. Pretrained weights are downloaded at runtime from
[`XiaomiRobotics/Xiaomi-Robotics-0-Pretrain`](https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-0-Pretrain);
its VLM is [Qwen3-VL](https://arxiv.org/abs/2511.21631).

```bibtex
@article{cai2026xiaomirobotics0,
  title   = {Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with
             Real-Time Execution},
  author  = {Cai, Rui and Guo, Jian and He, Xin and Jin, Peng and Li, Jie and Lin, Bin and
             Liu, Fei and Liu, Wei and Ma, Fan and Ma, Kai and Qiu, Feng and Qu, Hao and
             Su, Yang and Sun, Qi and Wang, Di and Wang, Dong and Wang, Yu and Wu, Rui and
             Xiang, Dong and Yang, Yi and Ye, Hao and Zhang, Yu and Zhou, Qiang},
  journal = {arXiv preprint arXiv:2602.12684},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.12684}
}
```

**[FullPart / PartVerse-XL](https://arxiv.org/abs/2510.26140)** — the source of the 3D part
data our datasets are built from.

```bibtex
@article{ding2025fullpart,
  title   = {FullPart: Generating Each 3D Part at Full Resolution},
  author  = {Ding, Lihe and Dong, Shaocong and Li, Yiming and Gao, Chenjian and Chen, Xiao and
             Han, Rui and Kuang, Yaoyan and Zhang, Hongxing and Huang, Boyang and
             Huang, Zhanpeng and Wang, Zhi and Xu, Dan and Xue, Tianfan},
  journal = {arXiv preprint arXiv:2510.26140},
  year    = {2025},
  url     = {https://arxiv.org/abs/2510.26140}
}
```

**[Objaverse-XL](https://objaverse.allenai.org/)** — the upstream 3D object collection
PartVerse-XL derives from.

```bibtex
@misc{deitke2023objaversexluniverse10m3d,
      title={Objaverse-XL: A Universe of 10M+ 3D Objects}, 
      author={Matt Deitke and Ruoshi Liu and Matthew Wallingford and Huong Ngo and Oscar Michel and Aditya Kusupati and Alan Fan and Christian Laforte and Vikram Voleti and Samir Yitzhak Gadre and Eli VanderBilt and Aniruddha Kembhavi and Carl Vondrick and Georgia Gkioxari and Kiana Ehsani and Ludwig Schmidt and Ali Farhadi},
      year={2023},
      eprint={2307.05663},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2307.05663}, 
}
}
```

Two methods the pipeline relies on directly:
[Set-of-Mark prompting](https://arxiv.org/abs/2310.11441) for the datasets' visual prompts,
and the [continuous 6D rotation representation](https://arxiv.org/abs/1812.07035) used by the
rotation head.

## License

Released under the [Apache License 2.0](LICENSE). Portions are derived from openpi, which is
also Apache 2.0; see [NOTICE](NOTICE) for the required attributions.
