# Generating Adversarial Examples with Adversarial Networks

[Paper](https://arxiv.org/pdf/1801.02610.pdf) | IJCAI 2018




## Usage

#### Inference
```bash
$ python3 advgan.py --img images/0.jpg --target 4 --model Model_C --bound 0.3
```  
Each of these settings has a separate Generator trained. This code loads appropriate trained model from ```saved/``` directory based on given arguments. As of now there are 22 Generators for different targets, different bounds (0.2 and 0.3) and target models (only ```Model_C``` for now).


#### Training AdvGAN (Untargeted)
```bash
$ python3 train_advgan.py --model Model_C --gpu
```  
#### Training AdvGAN (Targeted)
```bash
$ python3 train_advgan.py --model Model_C --target 4 --thres 0.3 --gpu
# thres: Perturbation bound 
```  
Use ```--help``` for other arguments available (```epochs```, ```batch_size```, ```lr``` etc.)


#### Training Target Models (Models A, B and C)
```bash
$ python3 train_target_models.py --model Model_C
```  

For TensorBoard visualization,
```bash
$ python3 generators.py
$ python3 discriminators.py
```  

This code supports only MNIST dataset for now. Same notations as in paper are followed (mostly).



## Results 
There are few changes that have been made for model to work.
* Generator in paper has ```ReLU``` on the last layer. If input data is normalized to [-1 1] there wouldn't be any perturbation in the negative region. As expected accuracies were poor (~10% Untargeted). So ```ReLU``` was removed. Also, data normalization had significat effect on performance. With [-1 1] accuracies were around 70%. But with [0 1] normalization accuracies were ~99%.
* Perturbations (```pert```) and adversarial images (```x + pert```) were clipped. It's not converging otherwise.

These results are for the following settings.
* Dataset - MNIST
* Data normalization - [0 1]
* thres (perturbation bound) - 0.3 and 0.2
* No ```ReLU``` at the end in Generator
* Epochs - 15
* Batch Size - 128
* LR Scheduler - ```step_size``` 5, ```gamma``` 0.1 and initial ```lr``` - 0.001


| Target     |Acc [thres: 0.3]  | Acc [thres: 0.2] |
|:----------:|:---------:|:---------:|
| Untargeted | 0.9921    | 0.8966    |    
| 0          | 0.9643    | 0.4330    |
| 1          | 0.9822    | 0.4749    |  
| 2          | 0.9961    | 0.8499    |
| 3          | 0.9939    | 0.8696    |  
| 4          | 0.9833    | 0.6293    |
| 5          | 0.9918    | 0.7968    |  
| 6          | 0.9584    | 0.4652    |
| 7          | 0.9899    | 0.6866    |  
| 8          | 0.9943    | 0.8430    |
| 9          | 0.9922    | 0.7610    |  





#### Untargeted
| <img src="images/results/untargeted_0_9.png" width="84"> | <img src="images/results/untargeted_1_3.png" width="84"> |<img src="images/results/untargeted_2_8.png" width="84"> | <img src="images/results/untargeted_3_8.png" width="84"> |  <img src="images/results/untargeted_4_4.png" width="84"> | <img src="images/results/untargeted_5_3.png" width="84"> | <img src="images/results/untargeted_6_8.png" width="84"> | <img src="images/results/untargeted_7_3.png" width="84"> | <img src="images/results/untargeted_8_3.png" width="84"> | <img src="images/results/untargeted_9_8.png" width="84"> | 
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|Pred: 9|Pred: 3|Pred: 8|Pred: 8|Pred: 4|Pred: 3|Pred: 8|Pred: 3|Pred: 3|Pred: 8|


#### Targeted
| Target: 0 | Target: 1 | Target: 2 | Target: 3 | Target: 4 | Target: 5 | Target: 6 | Target: 7 | Target: 8 | Target: 9 |  
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| <img src="images/results/targeted_0_0_0.png" width="84"> | <img src="images/results/targeted_0_1_1.png" width="84"> |<img src="images/results/targeted_0_2_2.png" width="84"> | <img src="images/results/targeted_0_3_3.png" width="84"> |  <img src="images/results/targeted_0_4_4.png" width="84"> | <img src="images/results/targeted_0_5_5.png" width="84"> | <img src="images/results/targeted_0_6_6.png" width="84"> | <img src="images/results/targeted_0_7_7.png" width="84"> | <img src="images/results/targeted_0_8_8.png" width="84"> | <img src="images/results/targeted_0_9_9.png" width="84"> |
|Pred: 0|Pred: 1|Pred: 2|Pred: 3|Pred: 4|Pred: 5|Pred: 6|Pred: 7|Pred: 8|Pred: 9|
| <img src="images/results/targeted_1_0_0.png" width="84"> | <img src="images/results/targeted_1_1_1.png" width="84"> |<img src="images/results/targeted_1_2_2.png" width="84"> | <img src="images/results/targeted_1_3_3.png" width="84"> |  <img src="images/results/targeted_1_4_4.png" width="84"> | <img src="images/results/targeted_1_5_5.png" width="84"> | <img src="images/results/targeted_1_6_6.png" width="84"> | <img src="images/results/targeted_1_7_7.png" width="84"> | <img src="images/results/targeted_1_8_8.png" width="84"> | <img src="images/results/targeted_1_9_9.png" width="84"> |
|Pred: 0|Pred: 1|Pred: 2|Pred: 3|Pred: 4|Pred: 5|Pred: 6|Pred: 7|Pred: 8|Pred: 9|
| <img src="images/results/targeted_9_0_0.png" width="84"> | <img src="images/results/targeted_9_1_1.png" width="84"> |<img src="images/results/targeted_9_2_2.png" width="84"> | <img src="images/results/targeted_9_3_3.png" width="84"> |  <img src="images/results/targeted_9_4_4.png" width="84"> | <img src="images/results/targeted_9_5_5.png" width="84"> | <img src="images/results/targeted_9_6_6.png" width="84"> | <img src="images/results/targeted_9_7_7.png" width="84"> | <img src="images/results/targeted_9_8_8.png" width="84"> | <img src="images/results/targeted_9_9_9.png" width="84"> |
|Pred: 0|Pred: 1|Pred: 2|Pred: 3|Pred: 4|Pred: 5|Pred: 6|Pred: 7|Pred: 8|Pred: 9|

    

