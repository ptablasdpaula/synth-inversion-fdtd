# NESS Guitar → Equivariant Flow Matching

Symmetry-aware synthesizer inversion for FDTD guitar synthesis using conditional flow matching with learnable equivariance (Param2Tok).

Based on [Hayes et al. (2025)](https://arxiv.org/abs/2506.07199) "Audio Synthesizer Inversion in Symmetric Parameter Spaces with Approximately Equivariant Flow Matching", adapted for physical modelling guitar synthesis via the [NESS framework](https://github.com/Edinburgh-Acoustics-and-Audio-Group/ness).

## Architecture overview

```
Audio (stereo WAV, 44.1kHz, 4s)
    ↓
Log-Mel Spectrogram (2 × 128 × 401)
    ↓
Audio Spectrogram Transformer (AST)
    ↓  8 learnt query tokens → per-layer conditioning
Conditional Vector Field (DiT + Param2Tok)
    ↓  40 permutable tokens, no positional encoding
Parameter Vector (1240-d)
    ↓  dereparameterise
Guitar Tab (up to 40 pluck events)
```

### Parameter vector layout

Each pluck event is a 31-dimensional vector:

| Dims  | Field          | Encoding                     |
|-------|----------------|------------------------------|
| 0     | exists         | binary (1.0 = real event)    |
| 1     | start_time     | continuous, scaled to [-1,1] |
| 2–7   | string_index   | one-hot (6 strings)          |
| 8–28  | fret_number    | one-hot (21 frets, 0–20)     |
| 29    | amplitude      | continuous, scaled to [-1,1] |
| 30    | pluck_duration | continuous, scaled to [-1,1] |

40 pluck slots × 31 dims = **1240 total parameters**.

### Symmetry structure

The 40 pluck slots are **permutation-symmetric**: reordering the event list produces identical audio from NESS. The Param2Tok module with `pe_type=none` (no positional encoding) makes the DiT vector field equivariant to these permutations, so the flow learns an invariant conditional density over unordered event sets.

Additionally, the guitar exhibits **quasi-symmetry**: the same pitch can be produced by different string/fret combinations (e.g. C4 on string 2 fret 1 vs string 3 fret 5). The generative (flow-based) approach handles this naturally by placing probability mass on multiple valid solutions.

## Workflow

### 1. Generate dataset

```bash
# On Andrena (or locally for small tests):
python generate_dataset.py \
    --output_dir ./data/ness_guitar \
    --num_samples 200000 \
    --ness_binary ./ness/ness-framework \
    --num_workers 32 \
    --seed 42

# Compute mel statistics for standardisation:
python compute_stats.py \
    --data_dir ./data/ness_guitar \
    --num_samples 10000
```

Each sample produces:
- `sample_XXXXXX.wav` — stereo mix at 44.1kHz, 4 seconds
- `sample_XXXXXX.pt` — reparameterised parameter tensor (1240-d)
- `sample_XXXXXX.json` — human-readable metadata

### 2. Train

```bash
# Param2Tok equivariant model (main):
python train.py experiment=ness_param2tok

# KSin baseline (fixed tokenisation):
python train.py experiment=ness_ksin_baseline

# Debug on CPU:
python train.py trainer=cpu logger=none data.batch_size=4 data.num_workers=0
```

### 3. Infer

```bash
python infer.py \
    --checkpoint outputs/.../checkpoints/last.ckpt \
    --audio_dir ./test_audio \
    --output_dir ./predictions \
    --num_samples 5 \
    --steps 100
```

This produces:
- Predicted parameter tensors (`.pt`)
- Human-readable JSON event lists
- NESS XML files for resynthesis

## Project structure

```
ness-flow/
├── train.py                    # Hydra entry point
├── infer.py                    # Inference script
├── generate_dataset.py         # Offline NESS data generation
├── compute_stats.py            # Mel statistics computation
├── configs/
│   ├── train.yaml              # Root config
│   ├── data/ness_guitar.yaml
│   ├── model/
│   │   ├── ness_flow.yaml      # Param2Tok (main)
│   │   └── ness_flow_ksin.yaml # KSin baseline
│   ├── experiment/
│   │   ├── ness_param2tok.yaml
│   │   └── ness_ksin_baseline.yaml
│   ├── callbacks/default.yaml
│   ├── trainer/{gpu,cpu}.yaml
│   ├── logger/{wandb,none}.yaml
│   └── paths/default.yaml
├── src/
│   ├── data/
│   │   └── ness_datamodule.py  # Dataset + DataModule
│   ├── models/
│   │   ├── transformer.py      # AST, DiT, Param2Tok (from Ben's repo)
│   │   └── ness_flow_module.py # Lightning module
│   └── utils/
│       ├── __init__.py         # Hydra helpers
│       └── callbacks.py
├── scripts/
│   ├── generate_data.sh        # SLURM data generation
│   └── train.sh                # SLURM training
└── .project-root
```

## Dependencies

```
torch >= 2.0
torchaudio
lightning >= 2.0
hydra-core >= 1.3
omegaconf
rootutils
wandb
einops
librosa
soundfile
tqdm
numpy
```

## Key differences from Hayes et al. (Surge XT)

| Aspect | Surge XT | NESS Guitar |
|--------|----------|-------------|
| Synthesizer | VST plugin (pedalboard) | C++ FDTD simulator |
| Parameters | 92–165 (oscillators, filters, LFOs) | 1240 (40 pluck events × 31) |
| Symmetry | Oscillator/filter/LFO permutations | Pluck event permutation + quasi-symmetry (same pitch from different fingerings) |
| Audio | 44.1kHz stereo, 4s | 48kHz → resampled to 44.1kHz stereo, 4s |
| Dataset size | 2M samples | Target: 200k (FDTD is slower) |
| Tokens | 128 (Param2Tok discovers grouping) | 40 (one per pluck slot, naturally permutable) |