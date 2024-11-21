
![Logo](https://ars.els-cdn.com/content/image/1-s2.0-S2666827021001067-gr6.jpg)

# Brain tumor segmentaion and classification

**Brain Tumor Detection and Classification Using MRI**  

This project focuses on the automated detection and classification of brain tumors using advanced image processing and machine learning techniques. Built in MATLAB, it combines classical segmentation methods (Otsu, K-means, Fuzzy C-means) with state-of-the-art deep learning models (ResNet-50, Inception v3) for precise tumor identification and classification. The pipeline includes preprocessing, feature extraction using GLCM and DWT, and evaluation using metrics such as accuracy and F1 score.  

Key Features:  
- **Segmentation**: Implements multiple algorithms like Watershed, K-means, and Fuzzy C-means for tumor localization.  
- **Feature Extraction**: Utilizes statistical and wavelet-based methods to capture essential image characteristics.  
- **Classification**: Employs pretrained deep learning models to distinguish between glioma, meningioma, pituitary tumors, and healthy cases.  
- **Performance Metrics**: Evaluates accuracy, precision, and recall to ensure robust results.  

This project is designed to aid medical professionals with an efficient, non-invasive diagnostic tool for early detection and improved patient outcomes. It aligns with SDG Goal 3: Good Health and Well-being.

## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)

### Features 
- **Comprehensive Segmentation**: Implements multiple methods like K-means clustering, Fuzzy C-means, and Watershed for precise tumor localization in MRI scans.  
- **Feature Extraction**: Utilizes Gray Level Co-occurrence Matrix (GLCM) and Discrete Wavelet Transform (DWT) for enhanced image feature analysis.  
- **Advanced Classification**: Leverages pre-trained models (ResNet-50, Inception V3) for multi-class tumor classification (glioma, meningioma, pituitary, and no tumor).  
- **Real-Time Visualization**: Includes visual outputs for segmentation and classification results, aiding interpretability.  
- **Performance Metrics**: Evaluates models using metrics like accuracy, precision, recall, F1-score, and confusion matrices for robust validation.  

### Project Workflow  
1. **Data Preprocessing**: MRI images are normalized and augmented to improve model generalization.  
2. **Segmentation**: Various algorithms isolate tumor regions effectively for subsequent analysis.  
3. **Feature Extraction**: Captures texture and structural features essential for tumor differentiation.  
4. **Classification**: Deep learning models classify the extracted regions into predefined tumor categories.  
5. **Evaluation**: Performance is assessed through metrics and visualization for quality assurance.

### References  
1. [Brain Tumor Detection Using Machine Learning and CNN](https://github.com/Sadia-Noor/Brain-Tumor-Detection-using-Machine-Learning-Algorithms-and-Convolutional-Neural-Network) – Combines ML algorithms and CNN for tumor detection with detailed documentation.  
2. [Deep Learning for Brain Tumor Classification](https://github.com/uniyalmanas/Brain-Tumor-detection-using-Deep-Learning) – Explores CNN architectures like ResNet-50 and DenseNet-121 for brain tumor analysis.  
3. [Brain Tumor Classification with CNN](https://github.com/aanyaG8/Brain_Tumor_Classification_with_CNN) – Focuses on classification using CNNs and provides scripts for model deployment.  
4. [Brain Tumor Image Segmentation and Classification](https://github.com/andylow-wl/BrainTumorDetection) – Includes segmentation (UNet-VGG16) and classification models for enhanced medical imaging.  

These repositories provide additional insights and complementary approaches to enhance your project documentation and implementation.
