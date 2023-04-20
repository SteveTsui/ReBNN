# Resilient Binary Neural Network (ReBNN)
Pytorch implementation of our paper ["Resilient Binary Neural Network"](https://arxiv.org/abs/2302.00956) accepted by AAAI2023 as oral presentation.
## Tips

Any problem, please contact the first author (Email: shengxu@buaa.edu.cn). 

Our code is heavily borrowed from ReActNet (https://github.com/liuzechun/ReActNet).
## Dependencies
* Python 3.8
* Pytorch 1.7.1
* Torchvision 0.8.2

## ReBNN with two-stage tranining

We test our ReBNN using the same ResNet-18 structure and training setttings as [ReActNet](https://github.com/liuzechun/ReActNet), and obtain 66.9% top-1 accuracy.

| Methods | Top-1 acc | Top-5 acc | Quantized model link |Log|
|:-------:|:---------:|:---------:|:--------------------:|:---:|
|[ReActNet](https://arxiv.org/abs/2003.03488) |  65.9     |  -     | [Model](https://github.com/liuzechun/ReActNet#models) |-|
| [ReCU](https://arxiv.org/abs/2103.12369)    |  66.4     |  86.5     | [Model](https://github.com/z-hXu/ReCU)        |-|
| [RBONN](https://arxiv.org/abs/2209.01542)    |  66.7     |  87.0     | [Model](https://github.com/SteveTsui/RBONN)        |-|
| [ReBNN](https://arxiv.org/abs/2302.00956)    |  66.9     |  87.1     | -       |-|


To verify the performance of our quantized models with ReActNet-like structure on ImageNet, please do as the following steps:
1. Finish the first stage training using [ReActNet](https://github.com/liuzechun/ReActNet).
2. Use the following command:
```bash 
cd 2_step2_rebnn 
bash run.sh
```

If you find this work useful in your research, please consider to cite:

```
@article{xu2023resilient,
  title={Resilient Binary Neural Network},
  author={Xu, Sheng and Li, Yanjing and Ma, Teli and Lin, Mingbao and Dong, Hao and Zhang, Baochang and Gao, Peng and Lv, Jinhu},
  journal={arXiv preprint arXiv:2302.00956},
  year={2023}
}
```
