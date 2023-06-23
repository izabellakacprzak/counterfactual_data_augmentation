# Counterfactual Data Augmentation
An evaluation and comparison of a novel method of counterfactual image data augmentation as a debiasing technique for predictive models.

## Datasets
Dataset wrappers for all setups used in the experiments
- datasets/perturbedMNIST.py (i.e. Morpho-MNIST) - modified version of the MNIST dataset [1] with additional thickness and intensity attributes
- datasets/coloredMNIST.py - modified version of the MNIST dataset with color applied to the foreground of each image
- datasets/chestXRay.py - subset of the Mimic CXR dataset of chest x-rays. We filter out all samples other than healthy patients and patients diagnosed with Pleural Effusion

## Tests
Contains multiple test and evaluation files
- tests/dimensionality_reduction.py - PCA + t-SNE dimensionality reduction and plotting
- tests/fairness_analysis - bias evaluation through fairness analysis 
- tests/fairness_bias_estimation.py - helper file for bias calculation from fairness analysis [2]
- tests/test_model.py - evaluation pipeline for standard metrics used in machine learning
- tests/visualise_cfs.py - helper file for geenrating plots of example images from used datasets

## Utils
Helper functions for datasets and introducing bias synthetically as well as evaluation
- utils/cf_utils.py - utils for cf generation
- utils/colors.py - methods for the generationg of coloredMNIST as introducing synthetic bias
- utils/evaluate.py - evaluation functions
- utils/params.py - all parameters used for training and data file names
- utils/perturbations.py - methods for the generationg of perturbedMNIST as introducing synthetic bias
- utils/utils.py - miscellaneous utils used throughout the project

## Experiment files
- classifier.py - implementation of ResNet-based and DenseNet-based classifiers with training/validation/testing code
- chestxray.py - bias mitigation comparison experiments for Mimic CXR
- colored_mnist.py - bias mitigation comparison experiments for coloredMNIST
- perturbed_mnist.py - bias mitigation comparison experiments for perturbedMNIST

## References
[1] Yann LeCun and Corinna Cortes. The mnist database of handwritten digits. 2003. URL http://yann.lecun.com/exdb/mnist/ \
[2] Saloni Dash, Vineeth N Balasubramanian, and Amit Sharma. Evaluating and mitigating bias in image classifiers: A causal perspective using counterfactuals. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 915–924, 2022