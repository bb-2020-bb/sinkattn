# SinkAttention

Train-free SinkAttention calibration and inference code for Wan video generation
and Z-Image-Turbo image generation.

Model weights, generated masks, generated media, benchmark outputs, and external
CUDA backend checkouts are not included in this repository.

## Requirements

Use one Python environment for both routes. Install a PyTorch wheel that matches
your CUDA setup, then install the repository dependencies:

```bash
conda create -n sinkattention python=3.11 -y
conda activate sinkattention
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

Dense inference and offline calibration use `requirements.txt`. Sink inference
requires the external `Block-Sparse-Attention` CUDA backend:

```bash
git clone https://github.com/mit-han-lab/Block-Sparse-Attention.git
cd Block-Sparse-Attention
git submodule update --init csrc/cutlass
pip install packaging ninja
python setup.py install
python -c "import block_sparse_attn"
```

For Z-Image, use `diffusers` and `transformers` versions that expose
`ZImagePipeline`, `ZImageTransformer2DModel`, and `Qwen3Model`.

## Models

Prepare local Diffusers-format checkpoints:

```bash
export WAN_MODEL=/path/to/Wan2.1-T2V-1.3B-Diffusers
export ZIMG_MODEL=/path/to/Z-Image-Turbo
```

## Wan

Default protocol: `832x480`, `81` frames, `50` denoising steps, guidance scale
`5.0`.

```bash
bash wanx/scripts/run_sink_calibration.sh 0 \
  --model_path "$WAN_MODEL" \
  --output_mask_path /path/to/wan_sink_mask_cov85.pt

bash wanx/scripts/run_sink_inference.sh 0 \
  --model_path "$WAN_MODEL" \
  --attn_mode sink \
  --sink_mask_path /path/to/wan_sink_mask_cov85.pt
```

## Z-Image

Default protocol: `2048x2048`, `8` denoising steps, guidance scale `0.0`.

```bash
bash zimg/scripts/run_sink_calibration.sh \
  --model_path "$ZIMG_MODEL" \
  --output_mask_path /path/to/zimg_sink_mask_cov85.pt

bash zimg/scripts/run_sink_inference.sh \
  --model_path "$ZIMG_MODEL" \
  --attn_mode sink \
  --sink_mask_path /path/to/zimg_sink_mask_cov85.pt
```

## Notes

- Sink mask construction uses row-wise direct mean coverage with default
  threshold `0.85`.
- Dense inference and offline calibration do not require `block_sparse_attn`;
  `--attn_mode sink` does.
- Generated outputs under `wanx/outputs/`, `zimg/outputs/`, `outputs/`, and
  `results/` are ignored by default.
