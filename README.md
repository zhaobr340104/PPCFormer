Implementation of **PPCFormer: A Phase-Preserving Cell Transformer for Multispectral Filter Array Demosaicing**.

## Environment
Use Python 3.10+ and PyTorch 2.6+.

## Data
Obtain [CAVE](https://cave.cs.columbia.edu/repository/Multispectral) and [Chikusei](https://naotoyokoya.com/Download.html) from their providers.

## Training
python train.py --dataset CAVE --data-root data/CAVE --output runs/CAVE/seed_0 --seed 0 --device cuda
python train.py --dataset Chikusei --data-root data/Chikusei --output runs/Chikusei/seed_0 --seed 0 --device cuda

## Testing
python test.py --dataset CAVE --data-root data/CAVE --checkpoint runs/CAVE/seed_0/best_val.pth --output results/CAVE --device cuda --save-cubes
python test.py --dataset Chikusei --data-root data/Chikusei --checkpoint runs/Chikusei/seed_0/best_val.pth --output results/Chikusei --device cuda --save-cubes
