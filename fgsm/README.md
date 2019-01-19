# Fast Gradient Sign Method

[Paper](https://arxiv.org/abs/1412.6572)  




## Usage

* **Run the script**
```bash
$ python3 fgsm_mnist.py --img one.jpg --gpu
```  

```bash
$ python3 fgsm_imagenet.py --img goldfish.jpg --model resnet18 --gpu
```  

```fgsm_mnsit.py``` - for attack on custom model trained on MNIST whose weights are ```9920.pth.tar```.  
```fgsm_imagenet``` - for pretrained imagenet models - resnet18, resnet50 etc.


* **Control keys**  
  - use trackbar to change `epsilon` (max norm)  
  - `esc` - close  
  - `s` - save perturbation and adversarial image  


## Demo    
![fgsm.gif](images/demo/fgsm.gif) 


## Models
Dataset - **MNIST**  
Accuracy - **99.20%**

```
----------------------------------------------------------------
# Basic_CNN
"""
input   - (1, 28, 28)
block 1 - (32, 28, 28)
maxpool - (32, 14, 14)
block 2 - (64, 14, 14)
maxpool - (64, 7, 7), reshape to (7*7*64,)
fc1     - (7*7*64,) -> (200,)
fc2     - (200,) -> (10,)
"""

# block
Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
ReLU()
Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
BatchNorm2d(out_channels)
ReLU()

#
MaxPool2d(kernel_size=2, stride=2)

# fc
Linear(in, out)

----------------------------------------------------------------
```



## Results  
#### MNIST
| Adversarial Image | Perturbation | 
|:----:|:----:|   
| <img src="images/results/adv_4.png" width="84"> | <img src="images/results/perturbation_4_38.png" width="84"> |  
| Pred: 4 | eps: 38 |  
| <img src="images/results/adv_7.png" width="84"> | <img src="images/results/perturbation_7_60.png" width="84"> |  
| Pred: 7 | eps: 60 |   
| <img src="images/results/adv_8(2).png" width="84"> | <img src="images/results/perturbation_8(2)_42.png" width="84"> |  
| Pred: 8 | eps: 42 |  
| <img src="images/results/adv_8.png" width="84"> | <img src="images/results/perturbation_8_12.png" width="84"> |  
| Pred: 8 | eps: 12 |    
| <img src="images/results/adv_9.png" width="84"> | <img src="images/results/perturbation_9_17.png" width="84"> |  
| Pred: 9 | eps: 17 |    

