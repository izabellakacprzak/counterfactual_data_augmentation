# Counterfactual Data Augmentation
An evaluation and comparison of a novel method of counterfactual image data augmentation as a debiasing technique for predictive models.

## Datasets
Dataset wrappers for all setups used in the experiments
- perturbedMNIST (i.e. Morpho-MNIST) - modified version of the MNIST dataset [1] with additional thickness and intensity attributes.
- coloredMNIST - modified version of the MNIST dataset with color applied to the foreground of each image.
- chestXRay - subset of the Mimic CXR dataset of chest x-rays. We filter out all samples other than healthy patients and patients diagnosed with Pleural Effusion.

## Tests
Contains multiple test and evaluation files
- dimensionality_reduction - PCA + t-SNE dimensionality reduction and plotting
- fairness analysis - bias evaluation through fairness analysis 
- fairness_bias_estimation - helper file for bias calculation from fairness analysis [2]
- test_model - evaluation pipeline for standard metrics used in machine learning
- visualise_cfs - helper file for geenrating plots of example images from used datasets

## Utils
Helper functions for datasets and introducing bias synthetically as well as evaluation
- cf_utils - utils for cf generation
- colors - methods for the generationg of coloredMNIST as introducing synthetic bias
- evaluate - evaluation functions
- params - all parameters used for training and data file names
- perturbations - methods for the generationg of perturbedMNIST as introducing synthetic bias
- utils - miscellaneous utils used throughout the project

## Main files
- classifier - implementation of ResNet-based and DenseNet-based classifiers with training/validation/testing code
- chestxray - bias mitigation comparison experiments for Mimic CXR
- colored_mnist - bias mitigation comparison experiments for coloredMNIST
- perturbed_mnist - bias mitigation comparison experiments for perturbedMNIST

## References
[1] Yann LeCun and Corinna Cortes. The mnist database of handwritten digits. 2003. URL http://yann.lecun.com/exdb/mnist/ \
[2] Saloni Dash, Vineeth N Balasubramanian, and Amit Sharma. Evaluating and mitigating bias in image classifiers: A causal perspective using counterfactuals. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 915–924, 2022