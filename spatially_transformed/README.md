# Spatially Transformed Adversarial Examples
[Paper](https://arxiv.org/abs/1801.02612) | ICLR 2018  
For clarity refer [View Synthesis by Appearance Flow](https://people.eecs.berkeley.edu/~tinghuiz/papers/eccv16_appflow.pdf).


## Usage
```bash
$ python3 stadv.py --img images/1.jpg --target 7
```  
Requires OpenCV for real-time visualization.  


## Demo
![0_1](images/demo/0_1.gif) ![1_2](images/demo/1_2.gif) ![2_3](images/demo/2_3.gif) ![3_4](images/demo/3_4.gif) ![4_5](images/demo/4_5.gif) ![5_6](images/demo/5_6.gif) ![6_7](images/demo/6_7.gif) ![7_8](images/demo/7_8.gif) ![8_9](images/demo/8_9.gif) ![9_0](images/demo/9_0.gif)

## Results  
#### MNIST
Column index is target label and ground truth images are along diagonal. 
  
  
![tile](images/tile.png?raw=true)

