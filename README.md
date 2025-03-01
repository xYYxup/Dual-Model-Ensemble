# DeepFake Detection Using Dual-Model Ensemble Approach by Group Winking😉

## Overview

In the digital age, the proliferation of DeepFake technology poses significant challenges to media authenticity and public trust. To address this issue, we have developed an AI-centric solution that leverages deep learning techniques to detect facial DeepFake content. Our approach integrates methodologies from two renowned frameworks—[DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) and [MultiModal-DeepFake](https://github.com/rshaojimmy/MultiModal-DeepFake)—and employs a secondary screening mechanism to enhance detection accuracy.

## Features

- **Dual-Model Ensemble**: Combines the strengths of two distinct DeepFake detection models to improve robustness and accuracy.
- **Secondary Screening**: Utilizes a custom BinaryFusion neural network to re-evaluate and optimize the classification results from the ensemble models.
- **CSV-Based Workflow**: Ensures efficient data processing through seamless integration and handling of model outputs via CSV files.

## Methodology

1. **Data Preparation**: Utilize the [FaceForensics++](https://github.com/ondyari/FaceForensics) dataset, a comprehensive collection of real and manipulated facial videos, for training and evaluation.
2. **Model Integration**:
   - **DeepfakeBench**: Implement state-of-the-art detection methods from DeepfakeBench, analyzing facial features and inconsistencies.
   - **MultiModal-DeepFake**: Employ multimodal detection techniques from MultiModal-DeepFake, assessing both visual and audio cues to identify DeepFakes.
3. **Secondary Screening with BinaryFusion**:
   - **Input**: Compile classification results from both models into a CSV file.
   - **BinaryFusion Network**: Apply a neural network to process the combined model outputs, conducting a secondary evaluation to enhance final classification accuracy.

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.8.0 or higher
- pandas 1.4.2 or higher
- For a complete list, refer to the `requirements.txt` file

### Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/xYYxup/deepfake-detection-ensemble.git
   cd deepfake-detection-ensemble
   ```
   
2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Dataset**:

   - Download the FaceForensics++ dataset and organize it according to the guidelines provided in DeepfakeBench.

4. **Configure Models**:

   - Prepare the detection models following the setup instructions in the DeepfakeBench and MultiModal-DeepFake repositories.

## Usage

1. **Run Detection Models**:

   - Execute the detection scripts from DeepfakeBench and MultiModal-DeepFake to generate classification results. Ensure the outputs are saved in CSV format.

2. **Secondary Screening**:

   - Process the combined CSV results using the BinaryFusion network:

     ```bash
     python binary_fusion.py --input_csv combined_results.csv --output_csv final_predictions.csv
     ```

3. **Evaluate Results**:

   - Assess the performance of the ensemble approach using standard metrics such as accuracy, precision, recall, and F1 score.

## Contribution

We welcome contributions from the community. Please follow these guidelines:

- Fork this repository and create a new branch for your feature or bug fix.
- Ensure your code adheres to existing styles and includes appropriate documentation.
- Submit a pull request with a clear description of your changes.

## Acknowledgements

We express our gratitude to the authors and contributors of DeepfakeBench, EfficientNetB4AttST, HAMMER and MultiModal-DeepFake for laying the groundwork for DeepFake detection.

## References

- Yan, Z., Zhang, Y., Yuan, X., Lyu, S., & Wu, B. (2023). DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection. *arXiv preprint arXiv:2307.01426*. [Link](https://github.com/SCLBD/DeepfakeBench)

- Shao, R., et al. (2020). MultiModal-DeepFake: A Multimodal Deep Learning Framework for Deepfake Detection. GitHub repository. [Link](https://github.com/rshaojimmy/MultiModal-DeepFake)

- N. Bonettini, E. D. Cannas, S. Mandelli, L. Bondi, P. Bestagini, and S. Tubaro, “Video Face Manipulation Detection Through Ensemble of CNNs,” Apr. 16, 2020, arXiv: arXiv:2004.07676. doi: 10.48550/arXiv.2004.07676.

- R. Shao, T. Wu, and Z. Liu, “Detecting and Grounding Multi-Modal Media Manipulation,” Apr. 05, 2023, arXiv: arXiv:2304.02556. doi: 10.48550/arXiv.2304.02556.



For more information, please visit our project repository.
