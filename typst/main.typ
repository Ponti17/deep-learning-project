#import "preamble.typ" as graceful-genetics
#import "@preview/physica:0.9.3"

#let frame(stroke) = (x, y) => (
  left: if x > 0 { 0pt } else { stroke },
  right: stroke,
  top: if y < 2 { stroke } else { 0pt },
  bottom: stroke,
)

#set table(
  fill: (_, y) => if calc.odd(y) { rgb("EAF2F5") },
  stroke: frame(rgb("21222C")),
)

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
      Melanoma is an aggressive form of skin cancer that contributes 1.4% of all cancer deaths in the US #cite(<nih>). Treatment of advanced melanoma is costly and potentially toxic #cite(<puma>). Research has shown a correlation between the presence of tumor infiltrating lymphocytes (TILs) and better responses to therapy. However, current methods for scoring TILs are manual, subjective and inconsistent. _PUMA_ is an open challenge for creating neural networks for classification of nuclei and tissue in melanoma. _PUMA_ consists of two _tracks_ with two _tasks_ each. Our submission was created for Track 1, Task 2: nuclei instance segmentation with classification of _Tumors_, _TILs_, and _other_. For source code see our #link("https://github.com/Ponti17/deep-learning-project")[GitHub (link)]. For more information on _PUMA_ see the official #link("https://puma.grand-challenge.org/")[page (link)].
  ],
)

= Previous Work
The classification of nuclei is a task that deep-learning computer vision techniques excel at. Models such as _Mask R-CNN_ and _NN192_ trained on large datasets such as _PanNuke_ #cite(<pannuke>), or _CoNSeP_ #cite(<hover>) generally give very good results for many segmentation and classification tasks. Melanoma nuclei, however, have the ability to mimick other cells, causing networks trained on previously mentioned datasets to misclassify #cite(<novel>).

= Dataset
To train a model to accurately segment and classify melanoma nuclei, a new dataset was required. The PUMA dataset was originally introduced in #cite(<novel>) and specifically made for melanoma. The public dataset includes:

- 103 primary and 103 metastatic melanoma regions of interest (ROI), scanned at 40x magnification with a resolution of 1024x1024 pixels.

- Context ROI of 5120x5120 pixels, centered around each ROI.

- Annotations for tissue and nuclei, provided by a medical professional and verified by a board-certified dermatopathologist.

#colbreak()

#figure(
  image("images/dataset_example.svg", width: 90%),
  placement: top,
  caption: [
    Dataset example.
  ],
) <dataset_example>

#figure(table(
  columns: (auto, 1fr, 1fr, 1fr),
  inset: 5pt,
  align: (x, _) => if x == 0 { left } else { right },
  table.header([], [*Total*], [*Primary*], [*Metastatic*]),
  [*Tumor*], [57,132], [22,579], [34,553],
  [*TIL*],  [22,141], [13,057], [9,084],
  [*Other*], [17,757], [13,337], [4,420],
  [*Sum*],   [97,030], [48,973], [48,057],
),
  caption: [
    Dataset nuclei count.
  ]
) <dataset_table>

As seen in #ref(<dataset_table>) there is significant class imbalance in the dataset. In addition the dataset is also smaller than other popular datasets such as _PanNuke_ which has over 200,000 labelled nuclei. The images are in ```.tif``` format which are easily parsed by packages such as ``` opencv``` and ``` pillow```. The annotations are in ```.geojson```, where the nuclei contours are described as polygon coordinates.

#pagebreak()

= Challenges
Nuclei segmentation and classification present some unique challenges that pose strict requirements to a good model candidate. During the project we have identified the following challenges.

- Separation of clustered nuclei.

- Classification of nuclei.

- Class imbalance.

= Hover-Net
To deal with the challenges presented, a HoVer-Net architecture was chosen (@hover-net-architecture). First proposed in #cite(<hover>), the HoVer-Net is a deep neural network designed purposefully for nuclei segmentation and classification. At the input, data is fed into a ResNet-50 #cite(<resnet>) encoder network. The ResNet-50 network is used for feature extraction. Following the ResNet are three independent decoder branches:

1. *Nuclear Pixel (NP) branch:* predicts whether or not a pixel belongs to a nuclei or the background.

2. *HoVer branch:* predicts the horizontal and vertical distances of nuclear pixels to their centers of mass.

3. *Nuclear Classification (NC) branch:* predicts whether or not a pixel belongs to a certain type of nuclei or the background.

Thus the NP and HoVer branches perform nuclear segmentation, where the NP branch is separating nuclear pixels from the background while the HoVer branch is separating touching nuclei. Finally the NC branch classifies each nuclear type.


It is important to note that the architecture of all three decoder branches are identical. They consist of several up-sampling layers followed by densely connected units.

The HoVer-Net is an attractive architecture for this type of problem. The HoVer branch effectively solves the problem of separating clustered nuclei. By using horizontal and vertical gradient maps (HoVer maps) (#ref(<data_conversion>)), the transition from one nuclei to another will result in a high derivative, making them easily separable. During inference the NP and HoVer maps will be used in conjunction to predict the moment (center), contours and bounding boxes of the nuclei. With the NC branch intended to solve the challenge of nuclear classification we are left with the challenge of _class imbalance_. We speculate that the impact of class imbalance can be lessened by careful choice and weighting of loss functions.

#figure(
  image("images/hovernet.png", width: 95%),
  placement: top,
  caption: [
    Overview of proposed HoVer-net architecture from #cite(<hover>).
  ],
) <hover-net-architecture>

#figure(
  image("images/data_fig.png", width: 95%),
  placement: top,
  caption: [
    Preprocessing of PUMA data into HoVer-Net targets.
  ],
) <data_conversion>

==  Loss Function
HoVer-Net has four sets of weights: $w_0, w_1, w_2$ and $w_3$ from the ResNet, HoVer branch, NP branch and NC branch respectively. The weights are optimized jointly using a loss function $cal(L)$ defined as

$ cal(L) = underbrace(lambda_a cal(L)_a + lambda_b cal(L)_b, "HoVer Branch") + underbrace(lambda_c cal(L)_c + lambda_d cal(L)_d, "NP Branch") + underbrace(lambda_e cal(L)_e + lambda_f cal(L)_f, "NC Branch") $

The constants $lambda_a, ..., lambda_f$ are scalars that give weight to their associated loss function. Thus the performance of each branch is based on the sum of two different loss functions. Summarized in #ref(<loss_function_table>), we use four different loss functions, Mean Squared Error (MSE), Mean Squared Gradient Error (MSGE), Cross Entropy (CE) and Dice. MSGE is a fairly novel loss function that was introduced in the original HoVer-Net Paper #cite(<hover>).

#figure(
  placement: top,
  table(
  columns: (auto, auto, auto, 1fr),
  inset: 5pt,
  align: (horizon, horizon, horizon, left),
  table.header([*Branch*], [*Symbol*], [*Name*], [*Formula*]),
  [HoVer], $cal(L)_a$, [MSE], $1/n sum_(i=1)^n (p_i - Gamma_i)^2$,
  [HoVer], $cal(L)_b$, [MSGE], $1/m sum_(i in M) (Delta_x p_(i,x) - Delta_x Gamma_(i,x))^2 \ + 1/m sum_(i in M) (Delta_y p_(i,y) - Delta_x Gamma_(i,y))^2$,
  [NP], $cal(L)_c$, [CE], $- 1/n sum_(i=1)^N sum_(k=1)^K X_(i,k)(I) log Y_(i,k)(I)$,
  [NP], $cal(L)_d$, [Dice], $1 - (2 dot sum_(i=1)^N (Y_i (I) dot X_i (I)) + epsilon )/(sum_(i=1)^N Y_i (I) + sum_(i=1)^N X_i (I) + epsilon)$,
  [TP], $cal(L)_e$, [CE], $- 1/n sum_(i=1)^N sum_(k=1)^K X_(i,k)(I) log Y_(i,k)(I)$,
  [TP], $cal(L)_f$, [Dice], $1 - (2 dot sum_(i=1)^N (Y_i (I) dot X_i (I)) + epsilon )/(sum_(i=1)^N Y_i (I) + sum_(i=1)^N X_i (I) + epsilon)$
),
  caption: [
    Loss functions for HoVer-Net.
  ]
) <loss_function_table>

#pagebreak()

= Training Strategy
Due to the size of the HoVer-Net it was quite compute intensive to train. To overcome this we used our access to the LUMI supercomputer, offering us an approximate speedup of over 3000% (yes thousand) compared to Andreas' RTX 2080.

To monitor the training process, relevant configuration parameters and training/validation statistics were logged using ``` Neptune.ai```.

All training instances were trained for 50 epochs as the loss and dice scores had all evened out completely at that point.

== Dataset
The data was split into a train, validation and test set of 70%, 15% and 15% respectively. Special attention was made to ensure that we have an even percentage of primary and metastatic ROIs in each split. Otherwise we could risk a scenario where the training set would contain a skewed amount of either primary or metastatic ROIs, leading to disappointing validation and test performance.

== Data Preprocessing
To extend the training set we used extensive data augmentation. Each image during training was randomly cropped to a size of 256x256 before being sent into our augmentation pipeline that applies:

- Horizontal and/or vertical flip.

- Blur (Gaussian or Median) or noise (Gaussian).

- Hue saturation shift.

- Brightness and contrast shift.

== Hyperparameters
Several hyperparameters had significant impact on model performance and were of interest to optimize. We created a standalone train loop and used ``` RayTune``` in conjunction with LUMIs computational power to find the best configuration. All runs were using the Adam optimizer with optional StepLR scheduling. Other than the usual hyperparameters, we were also particularly keen to test following:

- *Freezing the ResNet*: If we freeze the ResNet during the first $x$ epochs, we force the network to train the decoders. Later we unfreeze the ResNet and fine-tune everything with a lowered learning rate.

- *Pre-trained ResNet*: In the original HoVer-Net paper they used a pre-trained Preact ResNet-50. We wanted to experiment with the PyTorch ResNet-50 model as well as with a ResNeXt model proposed in #cite(<resnext>).

== Evaluation
During training we monitored closely the loss functions of all decoder branches and used the sum as an overall score of how well the model performs at the training set. For validation we calculate the Dice coefficient, given by

$ "NP Dice" = (2|P sect T|)/(|P| + |T|) $

To evaluate the classification performance, we also calculate a per-class Dice coefficient given by the intersection of pixels predicted as some class $P_C$ and the ground truth of pixels as that class $T_C$ over the sum.

$ "Per Class Dice" = (2|P_C sect T_C|)/(|P_C| + |T_C|) $

== Results
Unfortunately neither freezing the ResNet nor using ResNeXt resulted in any noticable performance difference. The network performance did however improve from hyperparameter optimization.

In particular the optimization of the loss function weights $lambda_a, ..., lambda_f$ improved the per-class dice loss. After 50 epochs all validation metrics have converged, after which the weights were extracted and saved. Type predictions of an image from the test set is shown in #ref(<predictions>). On the test set the model boasts a NP dice score of $0.886$ and a pixelwise accuracy of $0.943$.

#figure(
  image("images/prediction.svg", width: 90%),
  placement: bottom,
  caption: [
    Type predictions versus ground truth.
  ],
) <predictions>

#figure(
  placement: bottom,
  table(
  columns: (1fr, 1fr, auto, auto),
  inset: 5pt,
  align: (left, left, right, right),
  table.header([*Parameter*], [*Symbol*], [*Default*], [*Optimized*]),
  [HV MSE weight], $lambda_a$, [1], [1.5],
  [HV MSGE weight], $lambda_b$, [1], [1.9],
  [NP CE weight], $lambda_c$, [1], [0.56],
  [NP Dice weight], $lambda_d$, [1], [1.29],
  [NC CE weight], $lambda_e$, [1], [1.46],
  [NC Dice weight], $lambda_f$, [1], [1.52],
  [learning rate], [lr], [1e-4], [2.75e-4],
  [weight decay], [-], [1e-5], [3.1e-5],
  [scheduler step size], [-], [-], [16],
  [scheduler gamma], [-], [-], [4.7e-5],
),
  caption: [
    Optimized hyperparameters.
  ]
) <hyperparameters>

#pagebreak()

#bibliography("bibliography.yml", style: "ieee", title: "References")
