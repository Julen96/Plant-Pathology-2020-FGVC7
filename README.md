# Plant-Pathology-2020-FGVC7



## Introduction
This repository has all the material used for the [Plant Pathology 2020 - FGVC7](https://www.kaggle.com/c/plant-pathology-2020-fgvc7) Kaggle competition.

In order to do my best and obtain the maximum possible accuracy in the competition, several papers are going to be used, and their ideas are going to be implemented in the final model.

## Problem
The FGVC problem is notably characterized by its two intriguing properties, significant inter-class similarity and intra-class variations. Therefore, the small details will be the ones deciding if a picture belongs to a class or to anohter.

In order to focus into those small details, an attention mechanism will be needed. Here two approaches with prooved state of the art performance are presented.

## 1st Approach: Batch Confussion Norm
The BCN technique improves the FGVC learning by imposing class prediction confusion on each training batch, and consequently alleviate the possible overfitting due to exploring image feature of fine details. In addition, this method is implemented with an attention gated CNN model, boosted by the incorporation of Astrous Spatial Pyramid Pooling (ASPP) to extract discriminative features and proper attentions.

This methodology (BCN) is generic, for more effective learning of a DNN (Deep Neural Network) model to the underlying FGVC problem.


## 2nd Approach: 
State-of-the-art approaches typically involve a backbone CNN such as ResNet or VGG that is extended by
a method that localizes and attends to specific discriminative regions. This method aims to improve the
performance of a given backbone CNN with little increase in complexity and requiring just a single pass
through the backbone network. Specifically, the following three steps are proposed:

- __Localization module:__ In order to avoid having to rely on human bounding box annotations, an
efficient bounding box detector is trained and applied before the image is processed by the
backbone CNN. This localization module is lightweight and trained using only the class labels.

- __Global k-max pooling:__ This two-step pooling procedure first applies k-max pooling at the last
convolutional layer which is followed by an averaging operation over the K selected maximal
values in each feature map. This could be seen as a very simple form of attention.

- __Embedding layer:__ An embedding layer is inserted into the backbone CNN as penultima layer,
which is trained with a loss function composed of two parts: within-class loss and between-class
loss






