# Electronics Register Classification Using CNN and YOLOv11

## Project Overview

This project aims to classify electronic components based on their resistance values using deep learning algorithms. The dataset consists of various types of resistors labeled as **12k-ohm**, **20k-ohm**, **45k-ohm**, **68k-ohm**, and **73k-ohm**. The classification model uses a combination of **YOLOv11**, **Custom CNN**, and **VGG16** architectures, with **YOLOv11** achieving the best performance in terms of accuracy.

### Key Features:
- **Dataset:** 5 classes of resistors with values: 12k-ohm, 20k-ohm, 45k-ohm, 68k-ohm, and 73k-ohm.
- **Algorithms Used:**
  - **YOLOv11 (You Only Look Once v11):** Best-performing algorithm for object detection and classification in this project.
  - **Custom CNN:** A convolutional neural network designed specifically for this task to classify resistor images.
  - **VGG16:** A pre-trained CNN model fine-tuned for the classification task.

## Algorithms Used

### YOLOv11 (You Only Look Once v11)
YOLOv11 is a state-of-the-art object detection model that processes images and detects objects in real-time. Unlike traditional algorithms, which analyze images in separate stages, YOLO predicts bounding boxes and class probabilities directly from the full image in one evaluation. It is fast and effective, which makes it suitable for this project where the goal is to classify different resistor types from images. YOLOv11 performs better than the other models in this project due to its accuracy and efficiency in identifying small objects within images.

### Custom CNN (Convolutional Neural Network)
A **Custom CNN** is a deep learning model designed specifically for image classification. It consists of several layers, including convolutional layers for feature extraction, pooling layers for dimensionality reduction, and fully connected layers for making predictions. The Custom CNN in this project was fine-tuned to process images of resistors, learning relevant features like shape and color that are indicative of each resistor class.

### VGG16 (Visual Geometry Group 16)
VGG16 is a deep CNN architecture that has been pre-trained on a large dataset like ImageNet. It consists of 16 layers and is known for its simplicity and effectiveness in image recognition tasks. In this project, VGG16 was fine-tuned on the resistor dataset to improve classification accuracy. Although it provides good results, it is outperformed by YOLOv11 in this particular application.

## Installation

To get started with the project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/rohit-gupta99939/Electronics-Register-classification.git
2. **Navigate to the project directory:**

   ```bash
   pip install -r requirements.txt

3. **Install the required dependencies:**

   Create a virtual environment (optional) and install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```
   The requirements.txt includes dependencies such as TensorFlow, Keras, YOLOv11, OpenCV, and other libraries used in this project.
## Dataset
The dataset used in this project consists of images of resistors categorized into five classes based on their resistance values:

- 12k-ohm
- 20k-ohm
- 45k-ohm
- 68k-ohm
- 73k-ohm
  
These images are labeled accordingly, and you can add more images to the dataset for training the model on different resistor types.

## Results

The YOLOv11 model has been found to perform the best in terms of accuracy and speed. Its real-time object detection capability ensures quick and accurate classification of resistors. The Custom CNN and VGG16 also provide good results but with slightly lower performance in comparison to YOLOv11.

## Future Improvements

- **Expand Dataset:** Adding more resistor types to the dataset will help improve the model's generalization.
- **Real-time Detection:** Implement real-time resistor classification using a camera and YOLOv11.
- **Model Optimization:** Further tune the model hyperparameters to achieve even better accuracy.
