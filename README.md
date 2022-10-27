# semantic_segmentation_pytorch
Implementation of semantic segmentation models with pytorch  

## model list

- Unet | [paper](https://arxiv.org/pdf/1505.04597v1.pdf)
- CGNet | [paper](https://arxiv.org/abs/1811.08201)
- ResNet50 + Deeplabv3plus | [paper](https://arxiv.org/pdf/1802.02611.pdf)
- SegNet | [paper](https://arxiv.org/abs/1511.00561)

---

## Installation

```python
git clone https://github.com/chaeyeongyun/semantic_segmentation_pytorch.git
cd semantic_segmentation_pytorch
pip install -r requirements.txt
```

---

## Train

```python
python train.py --model unet --config ./config/config.yaml
```

## Test

```python
python test.py --model unet --data_dir /data/test --save_dir ./test weights ./train/best_miou.pth 
```