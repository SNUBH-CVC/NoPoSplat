## Installation
```
pip install git+https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git
cd pretrained_weights && wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
cd src/model/encoder/backbone/croco/curope/ && python setup.py build_ext --inplace
```

## Training
```
python -m src.main +experiment=dl3dv
```
[re10k_1x8](config/experiment/re10k_1x8.yaml)에 1 A6000 GPU로 학습하는 예시 참고해서 학습

## Dataset
### RealEstate10K
[download script](https://github.com/yilundu/cross_attention_renderer/tree/master/data_download)에서 pytube가 문제가 있어서 pytubefix로 변경하면 다운로드 된다. 

### DL3DV
```
cd datasets
# https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P 
git lfs install
git clone git@hf.co:datasets/DL3DV/DL3DV-ALL-480P
python src/scripts/unzip_dl3dv.py
python src/scripts/convert_dl3dv.py
```

## Analysis
- `encoder_noposplat.py`
- `backbone_croco.py`