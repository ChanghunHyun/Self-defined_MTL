# Regarding the repository
This repository contains the dataset used in our work “Multi-task Learning with Self-Defined Tasks for Adversarial Robustness of Deep Networks” published in IEEE Access 2024. 
—see the full text [here] https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10409191

The dataset files were initially created for personal academic purpose but have been uploaded, considering their potential usefulness for other researchers in the field.
The dataset files contain manually generated labels of MNIST, CIFAR10, GTSRB, SVHN, and Tiny-ImageNet for multi-task learning.
The auxiliary tasks were generated based on the built-in labels (identity labels) provided by the original dataset.
The auxiliary tasks were generated based on the visual or abstract characteristics of the data. 
Note that the manually generated labels may not accurately categorize each data sample. Especially the auxiliary tasks for Tiny ImageNet were generated with rough categorization.

Additionally, in our follow-up research, we propose a novel perturbation generation method that leverages auxiliary tasks, and the corresponding code is included here as well.

# Link to Data files
https://drive.google.com/drive/folders/10FVu_ap3EeIqCQJ2UxkBrdccSAd-7rHU?usp=sharing

# Regarding the labels of auxiliary tasks.
Below we describe which labels the auxiliary tasks in each dataset were assigned. The number in parentheses is the actual assigned label value in python.
1. MNIST
 - Auxiliary task 1 (odd or even) : odd(0), even(1)
 - Auxiliary task 2 (composite or prime) : composite number(0), prime number(1)
2. SVHN
 - Auxiliary task 1 (odd or even) : odd(0), even(1)
 - Auxiliary task 2 (composite or prime) : composite number(0), prime number(1)
3. GTSRB
 - Auxiliary task 1 (shape of traffic sign) : include circle(0), polygon(1)
 - Auxiliary task 2 (configuration of traffic sign) : symbol only(0), include character(1) 
4. CIFAR10
 - Auxiliary task 1 (animal or vehicle) : animal(0), vehicle(1)
 - Auxiliary task 2 (main activity area) : land(0), sky(1), water(2)
5. Tiny ImageNet
 - Auxiliary task 1 (rough categorization 1) : natural thing(0), artefact(1) 
 - Auxiliary task 2 (rough categorization 2) : animal(0), machine(1), others(2)

# How labels for auxiliary tasks are assigned based on the identity class?
For example, the MNIST dataset originally contains 10 digit classes from 0 to 9 as identity labels. Then, labels for two auxiliary tasks are assigned as below.
| identity                   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|:--------------------------:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| auxtask1 (odd-even)        | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 | 0 | 1 |
| auxtask2 (prime-composite) | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 0 | 0 |

The figure below illustrates the task generation mechanism proposed in (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10409191)

![그림1d](https://github.com/user-attachments/assets/06e06517-3951-4adc-8e2c-84a06c7cfa53)

This is a semi-formal approach requring user-established criteria(we denote it as a mapping function Pm in figure above) for regrouping the existing labels(identity) to generate new tasks.

# Follow-up Research: MP(Mixed Perturbations) for AT(Adversarial Training)
The auxiliary tasks provided in this repository also can be used for our perturbation generation method.
The figure below illustrates the mechanism of MP.

![그림1](https://github.com/user-attachments/assets/1960a1b9-5801-4223-96bb-305969899667)

MP generates perturbations using the gradient information from each task and then combines them via a weighted summation to produce the final perturbation. This approach diversifies the directions of the generated perturbations, yielding a richer set of adversarial examples for AT. Because MP fuses task-specific information at the final weighting stage, it also makes it straightforward to control and analyze the directionality of the final perturbation.




# Regarding License
The licenses for MNIST, CIFAR10, and GTSRB datasets explicitly state that users have the freedom to merge, modify, and distribute the datasets as they wish. SVHN and Tiny ImageNet datasets' licenses grant free usage only for non-commercial purposes. The dataset files uploaded in this repository are strongly recommended for academic purposes exclusively. Any violation of this recommendation or misuse of these datasets beyond academic use shall be the sole responsibility of the user.
