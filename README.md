# Transfer Learning for NLP

This repository contains code for transfer learning in Natural Language Processing (NLP) using transformer architecture. The main objectives of this project are:

1. Implement and pretrain a language model with a transformer architecture.
2. Use the pretrained model for transfer learning in a sentiment analysis task, where the goal is to classify book reviews into positive and negative categories.
3. Compare the performance of the pretrained model with a model trained from scratch.
<p align="center">
  <img src="https://github.com/RayaneKimo/Transfer-Learning/assets/85368764/eac5cc02-ab53-4c6f-8be5-c2c824187015" alt="2phases learning">
</p>



## Overview

The code is organized into several components:

1. **TransformerModel**: Defines the transformer-based language model. It includes an embedding layer, positional encoding, and a stack of transformer encoder layers.

2. **ClassificationHead**: Defines the classification head, which takes the output of the transformer model and produces predictions for the sentiment analysis task.

3. **Model**: Combines the language model and classification head to create the complete model for transfer learning.

4. **PositionalEncoding**: Implements positional encoding for the transformer model.

5. **Dataset**: Defines a custom dataset class for loading and processing the input data for both pretraining and sentiment analysis tasks.

6. **Training**: Contains functions for training the model on both language modeling and sentiment analysis tasks.

7. **Inference**: Includes functions for generating text using the pretrained model and making predictions on new data for sentiment analysis.

## Usage

1. **Pretraining the Language Model:**
   - Run the provided script to download the vocabulary file:
     ```bash
     !wget https://raw.githubusercontent.com/moussaKam/transfer_learning_transformers/main/dict.txt
     ```
   - Update the `path_vocab` variable in the code to point to the downloaded vocabulary file.
   - Execute the code for pretraining the language model.

2. **Text Generation using the Pretrained Model:**
   - Run the script to install the required sentencepiece library:
     ```bash
     !pip install sentencepiece
     ```
   - Download the sentencepiece model:
     ```bash
     !wget https://raw.githubusercontent.com/moussaKam/transfer_learning_transformers/main/sentencepiece.french.model
     ```
   - Update the `checkpoint` variable with the path to the pretrained model checkpoint.
   - Use the `infer_next_tokens` function to generate text based on an input prompt.

3. **Sentiment Analysis:**
   - Download the sentiment analysis dataset:
     ```bash
     !wget https://raw.githubusercontent.com/moussaKam/transfer_learning_transformers/main/cls-books/train.review.spm
     !wget https://raw.githubusercontent.com/moussaKam/transfer_learning_transformers/main/cls-books/train.label
     !wget https://raw.githubusercontent.com/moussaKam/transfer_learning_transformers/main/cls-books/test.review.spm
     !wget https://raw.githubusercontent.com/moussaKam/transfer_learning_transformers/main/cls-books/test.label
     ```
   - Update the paths to the training and validation datasets in the code.
   - Run the code for training the sentiment analysis model and evaluating its performance.

4. **Model Evaluation and Prediction:**
   - Update the paths to the pretrained model checkpoint and test data in the code.
   - Execute the script to predict sentiment labels for test data and evaluate the model's accuracy.

## Results and Sample Predictions

In the context of our model training, we compare both models, analyzing their respective performances and how they evolve across training epochs.

**Pretrained Model (Top accuracy: 80%):**
Initialized with weights learned from a vast amount of text data, the pretrained model begins with a higher accuracy compared to the second model. This initial advantage is expected since it carries knowledge from the pretrained weights. As the training progresses over multiple epochs, the accuracy of the model steadily improves, reaching a peak accuracy of 80%. This indicates that it has effectively leveraged the prior knowledge gained from pretraining, acquiring valuable features and representations from the large text corpus. This enhances its performance on the target task.

**From Scratch Model (Top accuracy: 77%):**
Conversely, the from scratch model starts with a lower accuracy compared to the pretrained model. This discrepancy arises from the fact that it begins with randomly initialized weights, devoid of any prior knowledge. As the training progresses, its accuracy improves, reaching a peak of 78%.

**Interpretation:**
The gap between the accuracy curves for both models clearly highlights the advantages of transfer learning using a pretrained model. The pretrained model consistently outperforms the from scratch model in terms of final accuracy. The higher initial accuracy reveals its capability to start with a head start in the fine-tuning task, as it has already acquired valuable features during pretraining, such as knowledge, semantics, and words polysemy. This demonstrates the importance of general natural language understanding.


## Conclusion

This repository provides a comprehensive implementation of transfer learning for NLP using transformer architecture. Users can leverage pretrained language models for downstream tasks, such as sentiment analysis, and achieve competitive results with minimal training. The code is designed to be easily customizable for different datasets and tasks within the NLP domain.
