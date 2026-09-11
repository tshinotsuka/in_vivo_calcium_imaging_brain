# 環境構築

```bash
conda create -n caimg python=3.11 -y
conda activate caimg

pip install torch torchvision          # CUDA 版。CPU なら:
                                       #   --index-url https://download.pytorch.org/whl/cpu
pip install suite2p pyyaml
pip install -e /media/tshino/DATA/Projects/in_vivo_water_imaging_brain

python -c "import suite2p, cellpose, torch; print(suite2p.__version__, torch.__version__, torch.cuda.is_available())"
```

- suite2p 1.1.0 は cellpose と torch を必須依存として引く（合計 ~5GB）
- `--torch-device` は installed torch に合わせる。CPU 版で `cuda` を渡すと落ちる
- `naming.py` / `scanimage_meta.py` は ivwib の単一正本。複製しない（roadmap §3.3）
- env は DATA 側に置く方針（再生成可能ゆえ。`conda config --append envs_dirs /media/tshino/DATA/envs`）
