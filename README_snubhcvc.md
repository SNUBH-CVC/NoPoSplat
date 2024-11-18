## Installation
```
cd pretrained_weights && wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
cd src/model/encoder/backbone/croco/curope/ && python setup.py build_ext --inplace
pip install git+https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git
```

## Training
```
python -m src.main +experiment=dl3dv
```
[re10k_1x8](config/experiment/re10k_1x8.yaml)에 1 A6000 GPU로 학습하는 예시 참고해서 학습

## Evaluation
- evaluation index 생성은 train과 별도로 해줘야 한다. `context`를 input으로, `target`을 그 사이의 image로 지정하는 방식
    - https://github.com/cvg/NoPoSplat/issues/14

```
python -m src.main +experiment=re10k mode=test dataset/view_sampler@dataset.re10k.view_sampler=evaluation dataset.re10k.roots=[datasets/re10k_subset] dataset.re10k.view_sampler.index_path=assets/evaluation_index_re10k_subset.json checkpointing.load=./pretrained_weights/re10k.ckpt test.save_image=true
```

## Export
- 3D GS를 ply 형식으로 저장한 뒤 Blender에서 확인할 수 있다. 
    - https://github.com/cvg/NoPoSplat/issues/10
    - https://www.kiriengine.app/blender-addon/3dgs-render

## Dataset

### RealEstate10K
[download script](https://github.com/yilundu/cross_attention_renderer/tree/master/data_download)에서 pytube가 문제가 있어서 pytubefix로 변경하면 다운로드 된다. 
python src/scripts/generate_realestate.py train

### DL3DV
```
cd datasets
# https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P 
git lfs install
git clone git@hf.co:datasets/DL3DV/DL3DV-ALL-480P
python src/scripts/unzip_dl3dv.py
python src/scripts/convert_dl3dv.py
```

### Coronary Angiogram
데이터 구조는 아래와 같다. 
```
cag/
    - {patient_id}_{study_date}_{left/right}
        - images/
            - frame_{frame_no:05d}.png
        - transforms.json (patient_id, study_date, frames[].file_path, frames[].intrinsic_matrix, frames[].angle_cls, frames[].series_no, frames[].frame_no)
```

데이터 변환 명령어
```python
python src/scripts/convert_cag.py
```
