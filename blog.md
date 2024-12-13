# Abstract
This is the final report for CS163 

## 1. Introduction

The goal of this project is to explore the robustness and practicality of the Oclusion removeal architecture in a constrained environment with limited computing resources and a downscaled dataset. For this project, we decided to aim specifically at the architecture presented by the paper "DesnowNet: Context-Aware Deep Network for Snow Removal". The goal is to achieve this with limited computing resources and a downscaled dataset. The purpose of the neural network is to remove the snow (noise) in snowy images and recover a snow-free image"The neural network is designed to remove snow (viewed as noise) from images and recover clear, snow-free visuals. While the original paper demonstrated strong results, we adapted the method to a scaled-down Snow100K dataset and evaluated its performance in a simplified setup.

Our work builds on the DesnowNet framework, focusing on understanding its ability to generalize under constraints. This includes analyzing the architecture’s modular components, adapting it for more efficient computation, and measuring performance on this reduced dataset. We also reflect on the challenges of balancing computational limits with maintaining model effectiveness. The rest of this paper is structured as follows:
- **Section II**: Background and Related Work
- **Section III**: Overview of the DesnowNet architecture.
- **Section IV**: Details of the dataset and scaling adaptations.
- **Section V**: Experimental results and observations.
- **Section VI**: Conclusions and reflections on the project.

## 2. Background and Related Work

### Understanding Image Restoration
Image restoration is a fundamental task in computer vision that aims to recover high-quality images from degraded inputs. Degradation can occur due to various factors, including noise, motion blur, glare, or occlusions. This process is crucial for applications where clear and accurate visuals are required, and it often serves as a pre-processing step for downstream tasks like image classification, object detection, or scene understanding.

In traditional machine learning pipelines, image restoration tasks are handled by minimizing a loss function that quantifies the difference between the restored image and the ground truth. Modern methods leverage deep learning models to capture complex patterns of degradation and restoration, achieving state-of-the-art results across various tasks.

### Importance of Image Restoration
Image restoration is not just a theoretical exercise but has significant practical implications. The quality of restored images directly impacts the performance of subsequent tasks in various applications, such as:
- **Autonomous Vehicles**: Ensuring robust navigation and obstacle detection under challenging weather or lighting conditions.
- **Security and Surveillance**: Enhancing the clarity of surveillance footage for better monitoring and incident analysis.
- **Photography and Media**: Providing tools for professional and consumer-grade image enhancement.
- **Scientific and Medical Imaging**: Enabling accurate analysis of degraded images in fields like astronomy, radiology, and microscopy.

Restoring degraded images ensures that critical information is preserved and can be reliably analyzed, regardless of the domain.

### Techniques in Image Restoration
Several specialized tasks fall under the umbrella of image restoration, each requiring unique methods and architectures:
- **Denoising**: Removing random noise introduced during image capture, such as sensor noise in low-light conditions.
- **Deblurring**: Correcting blur caused by motion, focus errors, or camera shake.
- **Occlusion Removal**: Addressing obstructions, such as rain streaks, snow, or dirt, that obscure parts of the image.
- **Super-Resolution**: Enhancing image resolution to reveal finer details, often beyond the original capture quality.

Traditional approaches to these problems often relied on hand-crafted features and assumptions about the degradation model (e.g., Gaussian noise for denoising). However, deep learning has revolutionized the field by enabling models to learn directly from data, resulting in more flexible and powerful solutions.

### Image Restoration as a Step in the Vision Pipeline
Image restoration plays a pivotal role in the broader vision pipeline. It acts as the bridge between raw image capture and high-level computer vision tasks such as:
- **Image Classification**: Assigning labels to objects or scenes in an image.
- **Object Detection**: Identifying and localizing objects within an image.
- **Semantic Segmentation**: Understanding the pixel-wise classification of scenes.

For instance, degraded images can severely impact the accuracy of classification models, making restoration a critical pre-processing step. This pipeline—moving from image capture to restoration and then analysis—ensures robustness in practical systems like autonomous vehicles and security networks.

### Deep Learning and Specialized Models
With the rise of deep learning, image restoration has evolved from task-specific models to modular architectures that can tackle multiple types of degradation. For example, models like convolutional neural networks (CNNs) and attention-based mechanisms have been adapted to handle specific challenges like translucency recovery and residual generation, as seen in architectures such as DesnowNet.

Despite their success, these models often face challenges in generalization and scalability, particularly when deployed in resource-constrained environments or with limited training data. Recent efforts in the field have focused on creating lightweight, adaptable architectures capable of maintaining performance under such constraints.

By combining these advancements with task-specific knowledge, modern image restoration methods continue to push the boundaries of what is possible, enabling more robust and versatile vision systems.
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
