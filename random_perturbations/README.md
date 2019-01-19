# Random Perturbations


From one of the first papers on Adversarial examples - [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572),
> The direction of perturbation, rather than the specific point in space, matters most. Space is
not full of pockets of adversarial examples that finely tile the reals like the rational numbers.  

This project examines this idea by testing the robustness of a DNN to randomly generated perturbations.



## Usage
```bash
$ python3 explore_space.py --img images/horse.png
```



## Demo
 ![fgsm.gif](images/horse_explore_demo.gif)  

This code adds to the input image (`img`) a randomly generated perturbation (`vec1`) which is subjected to a max norm constraint `eps`. This adversarial image lies on a hypercube centerd around the original image. To explore a region (a hypersphere) around the adversarial image (`img + vec1`), we add to it another perturbation (`vec2`) which is constrained by L<sub>2</sub> norm `rad`.  
Pressing keys `e` and `r` generates new `vec1` and `vec2` respectively.  




## Random Perturbations   
 
 The classifier is robust to these random perturbations even though they have severely degraded the image. Perturbations are clearly noticeable and have significantly higher max norm.  
 
 | ![horse_explore](images/horse_explore_single.gif) | ![automobile_explore](images/automobile_explore.gif) | ![truck_explore](images/truck_explore.gif) |  
 |:------------------------------------------:|:-----------------------:|:-----------:|  
 |             **horse**                      |      **automobile**     |: **truck** :|  
 
 In above images, there is no change in class labels and very small drops in probability.




## FGSM Perturbations  
A properly directed perturbation with max norm as low as 3, which is almost imperceptible, can fool the classifier.    

 | ![horse_scaled](images/horse_scaled.png) | ![horse_adversarial](images/horse_fgsm.png) | ![perturbation](images/horse_fgsm_pert.png) |
 |:---------:|:--------------------:|:--------------------------:|
 | **horse** |  predicted - **dog** | perturbation **(eps = 6)** |  
 

