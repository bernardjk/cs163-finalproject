# Abstract
The work is trying to replicate a desnowing method proposed by a paper. The proposed method uses modular neural networks to handle snow in different spatial frequencies, trajectories, and translucency.

## 1. Introduction
This work replicates a desnowing method proposed by the paper: *DesnowNet: Context-Aware Deep Network for Snow Removal*. The goal is to achieve this with limited computing resources and a downscaled dataset. The purpose of the neural network is to remove the snow (noise) in snowy images and recover a snow-free image. 

The rest of this paper is organized as follows:
- Section II provides the details of the proposed DesnowNet architecture.
- Section III elaborates on the dataset used for training.
- Section IV presents the experimental results.
- Section V concludes the work.

## 2. Proposed Method
The general mathematical equation is as follows:

\[ x = a \odot z + y \odot (1 - z) \tag{1} \]

Where:
- \( x \) is the snowy color image, a combination of the snow-free image \( y \) and a snow mask \( z \),
- \( \odot \) denotes element-wise multiplication, and
- \( a \) represents the chromatic aberration map.

To estimate the snow-free image \( \hat{y} \) from a given \( x \), we must also estimate the snow mask \( \hat{z} \) and the aberration map \( a \). The original paper used a neural network with two modules:
- Translucency Recovery (TR)
- Residual Generation (RG)

The relationship between \( \hat{y} \) is described as:

\[ \hat{y} = y' + r \tag{2} \]

### 2.1 Descriptor
Both TR and RG consist of Descriptor Extraction and Recovery submodules. Descriptors are extracted using an Inception-V4 network as a backbone and a subnetwork termed as the dilation pyramid (DP), defined as:

\[ f_t = \gamma \|_{n=0}^{n} B_{2n} (\Phi(x)) \tag{3} \]

Where:
- \( \Phi(x) \) represents the features from the last convolution layer of Inception-V4,
- \( B_{2n} \) denotes the dilated convolution with dilation factor \( 2n \).

### 2.2 Recovery Submodule
The recovery submodule of the TR module generates the estimated snow-free image by recovering the details behind translucent snow particles. It consists of:
1. Snow mask estimation (SE): Generates a snow mask \( \hat{z} \).
2. Aberration estimation (AE): Generates the chromatic aberration map for each RGB channel.

A new architecture, termed Pyramid Maxout, is used to select robust feature maps:

\[ M_{\beta}(f_t) = \text{max}( \text{conv}_1(f_t), \text{conv}_3(f_t), \dots, \text{conv}_{2\beta-1}(f_t) ) \tag{4} \]

The TR module recovers the content behind snow:

\[ y'_i = 
\begin{cases} 
\frac{x_i - a_i \times \hat{z}_i}{1 - \hat{z}_i}, & \text{if } \hat{z}_i < 1 \\
x_i, & \text{if } \hat{z}_i = 1 
\end{cases} \tag{5} \]

The RG module complements the residual \( r \) for improved image reconstruction:

\[ r = R_r(D_r(f_c)) = \sum_{\beta} \text{conv}_{2n-1}(f_r) \tag{6} \]

### 2.3 Loss Function
A loss network is constructed to measure losses at certain layers:

\[ L(m, \hat{m}) = \sum_{\tau=0}^{\tau} \| P_{2i}(m) - P_{2i}(\hat{m}) \|_2^2 \tag{7} \]

The overall loss function is defined as:

\[ L_{\text{overall}} = L_{y'} + L_{\hat{y}} + \lambda_{\hat{z}} L_{\hat{z}} + \lambda_w \| w \|_2^2 \tag{8} \]

Where:

- \( L_{\hat{z}} = L(z, \hat{z}) \),
- \( L_{\hat{y}} = L(y, \hat{y}) \),
- \( L_{y'} = L(y, y') \).

## 3. Dataset and Downscaling the Network
Due to limited computing resources and storage, a downscaled version of the Snow100K dataset was used, consisting of 10,000 images. The original neural network architecture was scaled down by a factor of 2 to enable training on Google Colab. Training was stabilized with a smaller number of epochs.

## 4. Results
Despite downscaling the network and dataset, the performance was acceptable. Larger, less-translucent snow particles remained in reconstructed images, but most minor details were recovered.

## 5. Conclusions
