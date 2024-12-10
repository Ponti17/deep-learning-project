#import "preamble.typ" as graceful-genetics
#import "@preview/physica:0.9.3"

#show: graceful-genetics.template.with(
  title: [Panoptic Segmentation of Nuclei in Advanced Melanoma],
  authors: (
    (
      name: "Andreas S. Pedersen",
      department: "Department of ECE",
      institution: "Aarhus University",
      city: "Aarhus",
      country: "Denmark",
      mail: "202104430@post.au.dk",
    ),
    (
      name: "Lukas N. Hedegaard",
      department: "Department of ECE",
      institution: "Aarhus University",
      city: "Aarhus",
      country: "Denmark",
      mail: "202108594@post.au.dk",
    ),
  ),
  date: (
    year: "2024",
    month: "December",
    day: "11",
  ),
  keywords: (
    "Deep-Learning",
    "AI",
    "Hover-Net",
    "Segmentation",
    "LUMI",
  ),
  abstract: [
      Melanoma is an aggressive form of skin cancer that contributes 1.4% of all cancer deaths in the US #cite(<nih>). Treatment of advanced melanoma is costly and potentially toxic #cite(<puma>). Research has shown a correlation between presence of tumor infiltrating lymphocytes (TILs) and better responses to therapy. However, current methods for scoring TILs are manual, subjective and inconsistent. _PUMA_ is an open challenge for creating neural networks for classification of nuclei and tissue in melanoma. For our submission we created a HoVer-net #cite(<hover>) for classifying nuclei. For source code and commit history, see our #link("https://github.com/Ponti17/deep-learning-project")[GitHub (link)].
  ],
)

= Task 
From the _PUMA_ challenge two possible tasks were given 

- *Task 1*: Semantic tissue segmentation of tumor, stroma, epithelium, blood vessel, and necrotic regions.
- *Task 2*: Nuclei detection for three classes; tumor, TILs (lymphocytes and plasma cells), and other cells (histiocytes, melanophages, neutrophils, stromal cells, epithelium, endothelium, and apoptotic cells).

In this paper we chose to work with *task 2*, meaning the goal is to make a model that detect the three classes of nuclei. 


= Dataset
The PUMA dataset was originally introduced in #cite(<MIDL>) and specifically made for Melanoma. It includes:

- 103 primary and 103 metastatic melanoma regions of interest (ROI), scanned at 40x magnification with a resolution of 1024x1024 pixels.

- Context ROI of 5120x5120 pixels, centered around each ROI.

- Annotations for tissue and nuclei, provided by a medical profession and verified by a board-certified dermatopathologist.



= Hover-Net
To deal with the unique challenges present in nuclei segmentation, a HoVer-Net architecture was chosen which can be seen on @hover-net-architecture. First proposed in #cite(<hover>), the HoVer-Net is a deep neural network consisting of a pre-trained residual network (ResNet) which extracts features from the input images. Following the ResNet are three distinct branches: 

1. *Nuclear Pixel (NP) branch:* predicts whether or not a pixel belongs to the nuclei or background.

2. *HoVer branch:* predicts the horizontal and vertical distances of nuclear pixels to their centers of mass.

3. *Nuclear Classification (NC) branch:* predicts the type of nucleus for each pixel.

Thus the NP and HoVer branches perform nuclear segmentation, where the NP branch is separating nuclear pixels from the background while the HoVer branch is separating touching nuclei. Finally the NC branch classifies each nuclear type.

#figure(
  image("images/hovernet.png"),
  placement: top,
  caption: [
    Overview of proposed HoVer-net architecture from #cite(<hover>).
  ],
) <hover-net-architecture>



==  Loss Function
HoVer-Net has four sets of weights: $w_0, w_1, w_2$ and $w_3$ from the ResNet, HoVer branch, NP branch and NC branch respectively. The weights are optimized jointly using a loss function $cal(L)$ defined as

$ cal(L) = lambda_a cal(L)_a + lambda_b cal(L)_b + lambda_c cal(L)_c + lambda_d cal(L)_d + lambda_e cal(L)_e + lambda_f cal(L)_f $

#set math.equation(numbering: "(1)")

, where $lambda_a and lambda_b$ represent the loss at the HoVer branch output, $lambda_c and lambda_d$ represent the loss at the output of the NP branch, and $lambda_e and lambda_f$ represent the loss at the output of the NC branch. 

The constants $lambda_a, ..., lambda_f$ are scalars that give weight to their associated loss function. The value of these scalars will be determined in a later section.

The multiple loss term computed as the output of the HoVer branch consists of the two loss function $lambda_a and lambda_b$. We denote $lambda_a$ as the mean-squared-error (MSE) between the predicted horizontal and vertical distances and the ground truth. We denote $lambda_b$ as computing the MSE between the horizontal and vertical gradients of the horizontal and vertical maps respectively and the corresponding gradients of the ground truth. 
Mathematically we define the functions as follow

$ cal(L_a) = 1/n sum_(i=1)^n p - g $

$ cal(L_b) = 1/m sum_(i in m) nabla_x p - nabla_x g $

, where $nabla_x and nabla_y$ denotes the gradients in the horizontal and vertical direction respectively. 

At the output of the NP and NC we compute the corss-entropy loss which correspond to $lambda_c and lambda_e$, and we compute the dice loss which correspond to $lambda_d and lambda_f$. Mathematically we define the functions as follow 

$ mono(C E) = - 1/n sum_(i=1)^N sum_(k=1)^K X_(i,k)(I) log Y_(i,k)(I) $

$ mono(D i c e) = 1 - (2 dot sum_(i=1)^N (Y_i (I) dot X_i (I)) + epsilon )/(sum_(i=1)^N Y_i (I) + sum_(i=1)^N X_i (I) + epsilon) $

ADD TEXT FOR THE MEANING OF LAST TWO FUNCTIONS!!!

= Training
For training the model we use 70% of the data while we allocated 15% for validation and 15% for testing. Furthermore, special attention was made to ensure that we have an even percentage of primary and metastatic ROIs in each dataset. This is to make sure that we did not end up with the training set primarily consisting of primary ROIs while the validation and test set consists of primarily of metastatic ROIs




== Data Processing

Given the rather small dataset we used extensive data augmentation to _extend_ the dataset before training. We use methods such as, 

- Reflect Padding #cite(<image-padding-medium>)

- Gaussian Blur

- Median Blur 

- Random Crop 

but not limited to.
The use of each augmentation is based on probabilities that can be found in the code.

== Raytune
In order to optimize the hyperparameters listed below, we implemented Raytune.

- Learning rate 

- Weight decay 

- Freeze

- Step size 

- Gamma

MAYBE EXPLAIN MORE HERE!!

== Neptune 
To log the loss functions for both training and validation we use Neptune. Through Neptune we are able to nicely log each of the six loss functions, the total loss and the validation loss metrics. It allows us to get an understanding of each of the loss function, which can be used to troubleshoot issues that might happen during training. 

#pagebreak()

= Results
#lorem(400)

#figure(
  placement: top,
  rect(width: 100%, height: 200pt, fill: gradient.linear(..color.map.rainbow)),
  caption: [
    PLACEHOLDER.
  ],
)

= Future Works
The ultimate greatest bottleneck in this project is the limited amount of data. However, we would be able to get more data but given the structure of the _Puma Challenge_ we would have to wait some time. Therefore, future work will include using the yet to be reveal data and strength the model.  

#pagebreak()

#bibliography("bibliography.yml", style: "ieee", title: "References")
