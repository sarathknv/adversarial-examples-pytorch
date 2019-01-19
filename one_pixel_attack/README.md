# One Pixel Attack for Fooling Deep Neural Networks

[Paper](https://arxiv.org/abs/1710.08864)  

> Existence of single pixel adversarial perturbations suggest that the assumption made in [Explaining and Harnessing Adversarial Examples](https://arxiv.org/pdf/1412.6572.pdf) that small additive perturbation on the values of many dimensions will accumulate and cause huge change to the output, might not be necessary for explaining why natural images are sensitive to small perturbations. 






## Usage

```bash
$ python3 one_pixel.py --img airplane.jpg --d 3 --iters 600 --popsize 10
```  
`d` is number of pixels to change (**L<sub>0</sub>** norm)  
`iters` and `popsize` are paprameters for [Differential Evolution](https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/)  






## Model
Dataset - **CIFAR-10**  
Accuracy - **85%**

```
----------------------------------------------------------------

"""
input   - (3, 32, 32)
block 1 - (32, 32, 32)
maxpool - (32, 16, 16)
block 2 - (64, 16, 16)
maxpool - (64, 8, 8)
block 3 - (128, 8, 8)
maxpool - (128, 4, 4)
block 4 - (128, 4, 4)
avgpool - (128, 1, 1), reshpe to (128,)
fc      - (128,) -> (10,)
"""

# block
Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
ReLU()
Conv2d(32, 32, kernel_size=3, padding=1)
BatchNorm2d(32)
ReLU()

#
MaxPool2d(kernel_size=2, stride=2)

# avgpool
AdaptiveAvgPool2d(1)

# fc
Linear(256, 10)

----------------------------------------------------------------
```




## Results  

Attacks are typically successful for images with low confidence. For successful attacks on high confidence images increase `d`, i.e., number of pixels to perturb.

| ![airplane](images/airplane_bird_8075.png) | ![bird](images/bird_deer_8933.png) | ![cat](images/cat_frog_8000.png)  |         ![frog](images/frog_bird_6866.png) |  ![horse](images/horse_deer_9406.png)  |
|:------------------------------------------:|:----------------------------------:|:---------------------------------:|:-----------------------------------------:|:--------------------------------------:|  
| **bird [0.8075]**                   |               **deer [0.8933]**           |  **frog [0.8000]**                |                        **bird [0.6866]**   |       **deer [0.9406]**                |









