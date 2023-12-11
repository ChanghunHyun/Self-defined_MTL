# Regarding the repository
The dataset files in this repository were initially created for personal academic purpose but have been uploaded, considering their potential usefulness for other researchers in the field.
The dataset files contain manually generated labels of MNIST, CIFAR10, GTSRB, SVHN, and Tiny-ImageNet for multi-task learning.
The auxiliary tasks were generated based on the built-in labels (identity labels) provided by the original dataset.
The auxiliary tasks were generated based on the visual or abstract characteristics of the data. 
Note that the manually generated labels may not accurately categorize each data sample. Especially the auxiliary tasks for Tiny ImageNet were generated with rough categorization.

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

Example of assigning labels for each auxiliary task based on identity.
MNIST has 10 digit classes from 0 to 9 as identity labels. The two auxiliary tasks are assigned as below.
- id class : 0 1 2 3 4 5 6 7 8 9
- auxtask1 : 0 1 0 1 0 1 0 1 0 1
- auxtask2 : 0 0 1 1 0 1 0 1 0 0

# Regarding License
The licenses for MNIST, CIFAR10, and GTSRB datasets explicitly state that users have the freedom to merge, modify, and distribute the datasets as they wish. SVHN and Tiny ImageNet datasets' licenses grant free usage only for non-commercial purposes. The dataset files uploaded in this repository are strongly recommended for academic purposes exclusively. Any violation of this recommendation or misuse of these datasets beyond academic use shall be the sole responsibility of the user.
