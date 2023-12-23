# AdaDFKD: Exploring Adaptive Inter-Sample Relationship in Data-Free Knowledge Distillation
Forked by a benchmark of data-free knowledge distillation from paper "AdaDFKD: Exploring Adaptive Inter-Sample Relationship in Data-Free Knowledge Distillation".
Forked by [CMI](https://arxiv.org/abs/2105.08584) and [CuDFKD](https://arxiv.org/abs/2208.13648).

## Installation
We use Pytorch for implementation. Please install the following requirement packages
```
pip install -r requirements.txt
```

## Our method
Based on curriculum learning and self-paced learning. Our method is called **AdaDFKD**. The result can be rephrased by scripts `scripts/adadfkd/`. For example, when distill ResNet18 from ResNet34 at the benchmark CIFAR10, please run the following script

```
bash scripts/adadfkd/adadfkd_cifar10_resnet34_resnet18.sh
```

The implementation is in `datafree/synthesis/adadfkd.py`.

## Result on CIFAR10
| Teacher    | Res34 | vgg11 | wrn-402 | wrn-402 | wrn-402 |
|------------|-------|-------|---------|---------|---------|
| Student    | Res18 | Res18 | wrn-401 | wrn-162 | wrn-161 |
| T. Scratch | 95.70 | 92.25 | 94.87   | 94.87   | 94.87   |
| S. Scratch | 95.20 | 95.20 | 93.94   | 93.95   | 91.12   |
| DAFL       | 92.22 | 81.10 | 81.33   | 81.55   | 72.15   |
| ZSKT       | 93.32 | 89.46 | 86.07   | 89.66   | 83.74   |
| ADI        | 93.26 | 90.36 | 86.85   | 89.72   | 83.01   |
| DFQ        | 94.61 | 90.84 | 91.69   | 92.01   | 86.14   |
| CMI        | 94.84 | 91.13 | 92.78   | 92.52   | 90.01  |
| CuDFKD     | 95.28 | 91.61 | 93.18   | 92.98   | 88.77   |
| AdaDFKD(G) | **95.32** | **92.19** | **93.38** | **93.15** | **90.05** |



## Result on CIFAR100

| Teacher    | Res34 | vgg11 | wrn-402 | wrn-402 |
|------------|-------|-------|---------|---------|
| Student    | Res18 | Res18 | wrn-401 | wrn-162 |
| T. Scratch | 78.05 | 71.32 | 75.83   | 75.83   |
| S. Scratch | 77.10 | 77.10 | 72.19   | 73.56   |
| DAFL       | 74.47 | 57.29 | 34.66   | 40.00   |
| ZSKT       | 67.74 | 34.72 | 29.73   | 28.44   |
| ADI        | 61.32 | 54.13 | 61.33   | 61.34   |
| DFQ        | 77.01 | 68.32 | 61.92   | 59.01   |
| CMI        | 77.02 | 70.56 | 66.89   | 65.11   |
| CuDFKD     | 75.87 | 71.22 | 66.43   | 65.94   |
| AdaDFKD    | **77.13** | **71.52**    | **68.19**   | **67.54** |


For Tiny-ImageNet, we provide the pretrained weights of as following:

| Model    | Acc@1 |
|----------|-------|
| [ResNet50](https://drive.google.com/file/d/1ujElJLOtQu3YBz7tHkRbb3T3e8W_pefs/view?usp=drive_link) | 67.18 |
| [ResNet34](https://drive.google.com/file/d/1Cze_g3SwIldwzh_jvXnGHf-SrC0kllDw/view?usp=sharing) | 62.43 |
| [ResNet18](https://drive.google.com/file/d/1XBLC9wewTzZusnZC6PdxmDuC-8Ns0lC_/view?usp=sharing) | 61.48 |

## Evaluation of the given paper
Given trained checkpoint, you can check the result of $Acc@1$, $Agree@1$ and $L_p$ by running the following jupyter notebooks:

- [Results for CIFARs](https://github.com/ljrprocc/DFKD/blob/main/notebooks/Eval_cifars.ipynb)
- [Results for Tiny ImageNet](https://github.com/ljrprocc/DFKD/blob/main/notebooks/Eval_Tiny_imagenet.ipynb)

## Other visualization results
Please refer to the paper. The file `notebooks/other_Visualization.ipynb` supports the majority of experimental results in the paper.

## Reference

* ZSKT: [Zero-shot Knowledge Transfer via Adversarial Belief Matching](https://arxiv.org/abs/1905.09768)
* DAFL: [Data-Free Learning of Student Networks](https://arxiv.org/abs/1904.01186)
* DeepInv: [Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion](https://arxiv.org/abs/1912.08795)
* DFQ: [Data-Free Network Quantization With Adversarial Knowledge Distillation](https://arxiv.org/abs/2005.04136)
* CMI: [Contrastive Model Inversion for Data-Free Knowledge Distillation](https://arxiv.org/abs/2105.08584)
* CuDFKD: [Dynamic Data-Free Knowledge Distillation by Easy-to-Hard Strategy](https://arxiv.org/abs/2208.13648)
* Fast10: [Up to 100x Faster Data-free Knowledge Distillation](https://arxiv.org/pdf/2112.06253.pdf)