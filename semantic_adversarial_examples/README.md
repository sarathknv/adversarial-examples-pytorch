# Semantic Adversarial Examples

>Images that are arbitrarily perturbed to fool the model, but in such a way that the modified image semantically represents the same object as the original image.
>
>  &mdash; <cite>Semantic Adversarial Examples, Hosseini et al., CVPRW 2018.</cite>

These attacks are discussed in the following papers:
1. [Semantic Adversarial Examples](https://arxiv.org/pdf/1804.00499.pdf) (CVPR Workshops 2018)
2. [Towards Compositional Adversarial Robustness: Generalizing Adversarial Training to Composite Semantic Perturbations](https://arxiv.org/pdf/2202.04235.pdf) (CVPR 2023) 

This repository implements **only single semantic attacks** discussed in [2]. The attacks are
constructed using Projected Gradient Descent (PGD) on the following components:
* Hue
* Saturation
* Rotation
* Brightness
* Contrast

See section 3.2. of [2] for the math behind these attacks.


## Dependencies  
* Python3  
* PyTorch
* [Kornia](https://github.com/kornia/kornia) (A differentiable computer vision library for PyTorch)
* OpenCV
* NumPy
* tqdm

## Contents

`attacks.py`: contains both gradient-based search and random search of the five semantic perturbation parameters.  
`main.py`: computes the robust accuracy of a model against single attacks. Need to manually change the model and the attack.  
`save_examples.py`: saves some adversarial images to disk, along with their original images.  
`examples`: contains some adversarial examples.  
`models`: VGG16 and ResNet models.  
`weights`: VGG16 and ResNet50 weights, trained on CIFAR-10.  
`visualizations`: codes to visualize the attacks.  



## Experiments 

Robust accuracy of single semantic attacks on ResNet50 trained on CIFAR-10.

|       | Clean |     Hue     | Saturation  |  Rotation   | Brightness  |  Contrast   |
|:------|------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|        
| Mine  | 92.72 |    81.65    |    92.37    |    88.49    |    90.04    |    91.40    |  
| Paper |  95.2 | 81.8 ± 0.0  | 94.0 ± 0.0  | 88.1 ± 0.1  | 92.1 ± 0.1  | 93.7 ± 0.1  |



## Examples

### Hue

|      Clean      |           <img src="examples/hue/6_y_1_y_pred_1.png" alt="img" width="128">            |            <img src="examples/hue/15_y_8_y_pred_8.png" alt="img" width="128">            |            <img src="examples/hue/20_y_7_y_pred_7.png" alt="img" width="128">            |           <img src="examples/hue/24_y_5_y_pred_5.png" alt="img" width="128">            |           <img src="examples/hue/25_y_2_y_pred_2.png" alt="img" width="128">            |
|:---------------:|:--------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|
|  **Perturbed**  | <img src="examples/hue/6_y_1_y_pred_1_y_adv_5_factor_1.836.png" alt="img" width="128"> | <img src="examples/hue/15_y_8_y_pred_8_y_adv_6_factor_-0.193.png" alt="img" width="128"> | <img src="examples/hue/20_y_7_y_pred_7_y_adv_9_factor_-1.913.png" alt="img" width="128"> | <img src="examples/hue/24_y_5_y_pred_5_y_adv_4_factor_3.142.png" alt="img" width="128"> | <img src="examples/hue/25_y_2_y_pred_2_y_adv_5_factor_2.274.png" alt="img" width="128"> |
| **Pred before** |                                       automobile                                       |                                           ship                                           |                                          horse                                           |                                           dog                                           |                                          bird                                           |
| **Pred after**  |                                          dog                                           |                                           frog                                           |                                          truck                                           |                                          deer                                           |                                           dog                                           |
|     **Hue**     |                                         1.836                                          |                                          -0.193                                          |                                          -1.913                                          |                                          3.142                                          |                                          2.274                                          |


&nbsp;
### Saturation

|      Clean      |           <img src="examples/saturation/213_y_9_y_pred_9.png" alt="img" width="128">            |           <img src="examples/saturation/412_y_3_y_pred_3.png" alt="img" width="128">            |           <img src="examples/saturation/434_y_3_y_pred_3.png" alt="img" width="128">            |           <img src="examples/saturation/468_y_7_y_pred_7.png" alt="img" width="128">            |           <img src="examples/saturation/776_y_2_y_pred_2.png" alt="img" width="128">            |
|:---------------:|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
|  **Perturbed**  | <img src="examples/saturation/213_y_9_y_pred_9_y_adv_8_factor_0.700.png" alt="img" width="128"> | <img src="examples/saturation/412_y_3_y_pred_3_y_adv_5_factor_1.156.png" alt="img" width="128"> | <img src="examples/saturation/434_y_3_y_pred_3_y_adv_1_factor_0.700.png" alt="img" width="128"> | <img src="examples/saturation/468_y_7_y_pred_7_y_adv_2_factor_1.159.png" alt="img" width="128"> | <img src="examples/saturation/776_y_2_y_pred_2_y_adv_5_factor_0.874.png" alt="img" width="128"> |
| **Pred before** |                                              truck                                              |                                               cat                                               |                                               cat                                               |                                              horse                                              |                                              bird                                               |
| **Pred after**  |                                              ship                                               |                                               dog                                               |                                           automobile                                            |                                              bird                                               |                                               dog                                               |
| **Saturation**  |                                              0.700                                              |                                              1.156                                              |                                              0.700                                              |                                              1.159                                              |                                              0.874                                              |


&nbsp;
### Rotation

|      Clean      |            <img src="examples/rotation/2_y_8_y_pred_8.png" alt="img" width="128">            |           <img src="examples/rotation/9_y_1_y_pred_1.png" alt="img" width="128">            |           <img src="examples/rotation/15_y_8_y_pred_8.png" alt="img" width="128">            |            <img src="examples/rotation/57_y_7_y_pred_7.png" alt="img" width="128">            |            <img src="examples/rotation/70_y_2_y_pred_2.png" alt="img" width="128">            |
|:---------------:|:--------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:|
|  **Perturbed**  | <img src="examples/rotation/2_y_8_y_pred_8_y_adv_1_factor_-6.318.png" alt="img" width="128"> | <img src="examples/rotation/9_y_1_y_pred_1_y_adv_3_factor_9.400.png" alt="img" width="128"> | <img src="examples/rotation/15_y_8_y_pred_8_y_adv_6_factor_0.383.png" alt="img" width="128"> | <img src="examples/rotation/57_y_7_y_pred_7_y_adv_3_factor_-9.586.png" alt="img" width="128"> | <img src="examples/rotation/70_y_2_y_pred_2_y_adv_3_factor_10.000.png" alt="img" width="128"> |
| **Pred before** |                                             ship                                             |                                         automobile                                          |                                             ship                                             |                                             horse                                             |                                             bird                                              |
| **Pred after**  |                                          automobile                                          |                                             cat                                             |                                             frog                                             |                                              cat                                              |                                              cat                                              |
|  **Rotation**   |                                            -6.318                                            |                                            9.400                                            |                                            0.383                                             |                                            -9.586                                             |                                            10.000                                             |


&nbsp;
### Brightness

|      Clean      |            <img src="examples/brightness/2_y_8_y_pred_8.png" alt="img" width="128">            |            <img src="examples/brightness/57_y_7_y_pred_7.png" alt="img" width="128">            |           <img src="examples/brightness/103_y_3_y_pred_3.png" alt="img" width="128">            |            <img src="examples/brightness/118_y_2_y_pred_2.png" alt="img" width="128">            |            <img src="examples/brightness/313_y_0_y_pred_0.png" alt="img" width="128">            |
|:---------------:|:----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|
|  **Perturbed**  | <img src="examples/brightness/2_y_8_y_pred_8_y_adv_1_factor_-0.200.png" alt="img" width="128"> | <img src="examples/brightness/57_y_7_y_pred_7_y_adv_5_factor_-0.150.png" alt="img" width="128"> | <img src="examples/brightness/103_y_3_y_pred_3_y_adv_6_factor_0.125.png" alt="img" width="128"> | <img src="examples/brightness/118_y_2_y_pred_2_y_adv_6_factor_-0.134.png" alt="img" width="128"> | <img src="examples/brightness/313_y_0_y_pred_0_y_adv_2_factor_-0.023.png" alt="img" width="128"> |
| **Pred before** |                                              ship                                              |                                              horse                                              |                                               cat                                               |                                               bird                                               |                                             airplane                                             |
| **Pred after**  |                                           automobile                                           |                                               dog                                               |                                              frog                                               |                                               frog                                               |                                               bird                                               |
| **Brightness**  |                                             -0.200                                             |                                             -0.150                                              |                                              0.125                                              |                                              -0.134                                              |                                              -0.023                                              |


&nbsp;
### Contrast

|      Clean      |           <img src="examples/contrast/2_y_8_y_pred_8.png" alt="img" width="128">            |           <img src="examples/contrast/57_y_7_y_pred_7.png" alt="img" width="128">            |           <img src="examples/contrast/70_y_2_y_pred_2.png" alt="img" width="128">            |           <img src="examples/contrast/103_y_3_y_pred_3.png" alt="img" width="128">            |           <img src="examples/contrast/140_y_6_y_pred_6.png" alt="img" width="128">            |
|:---------------:|:-------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------:|
|  **Perturbed**  | <img src="examples/contrast/2_y_8_y_pred_8_y_adv_1_factor_0.773.png" alt="img" width="128"> | <img src="examples/contrast/57_y_7_y_pred_7_y_adv_3_factor_0.836.png" alt="img" width="128"> | <img src="examples/contrast/70_y_2_y_pred_2_y_adv_3_factor_1.109.png" alt="img" width="128"> | <img src="examples/contrast/103_y_3_y_pred_3_y_adv_6_factor_0.700.png" alt="img" width="128"> | <img src="examples/contrast/140_y_6_y_pred_6_y_adv_3_factor_0.711.png" alt="img" width="128"> |
| **Pred before** |                                            ship                                             |                                            horse                                             |                                             bird                                             |                                              cat                                              |                                             frog                                              |
| **Pred after**  |                                         automobile                                          |                                             cat                                              |                                             cat                                              |                                             frog                                              |                                              cat                                              |
|  **Contrast**   |                                            0.773                                            |                                            0.836                                             |                                            1.109                                             |                                             0.700                                             |                                             0.711                                             |



## Citations

```bibtex
@inproceedings{hosseini2018semantic,
  title={Semantic adversarial examples},
  author={Hosseini, Hossein and Poovendran, Radha},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={1614--1619},
  year={2018}
}
```

```bibtex
@article{tsai2022towards,
  title={Towards compositional adversarial robustness: Generalizing adversarial training to composite semantic perturbations},
  author={Tsai, Yun-Yun and Hsiung, Lei and Chen, Pin-Yu and Ho, Tsung-Yi},
  journal={arXiv preprint arXiv:2202.04235},
  year={2022}
}
```
