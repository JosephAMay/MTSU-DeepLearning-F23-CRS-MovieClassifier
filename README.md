# MTSU-DeepLearning-F23-CRS-MovieClassifier
A demo of 2 different pretrained models (BERT and GPT2) and their ability to classify movie recommendations from the E-redial Dataset as good or bad based on cosine similarity. 

# Conversational Recommendation Quality Classification
## Overview
This repository focuses on the task of classifying conversational recommendations as either "good" or "bad." The main goal is to assess the quality of movie recommendations in conversations between a seeker and recommender using the E-redial dataset.

### Background
Conversational Recommender Systems (CRS) utilize conversations to recommend items to users. These systems consist of a conversation module aiming to extract relevant information efficiently and a recommendation module suggesting items to users based on gathered information. Interaction patterns in CRSs fall into four categories: System Active User Passive (SAUP), System Active User Engage (SAUE), System Active User Active (SAUA), and Active User Passive System (AUPS). The project addresses the gap in evaluating recommendation quality in conversational settings and aims to develop a network capable of classifying movie recommendations as good or bad.

Traditional metrics like BLEU, METEOR, and ROUGE assess machine-generated output quality but struggle with semantic nuances and paraphrases. Deep learning methods, leveraging word embeddings and transformer architectures like BERT and BART, capture semantic meaning effectively, but are not fine tuned for recommendation settings. The project utilizes BERT and GPT2 for embedding and classification tasks. Cosine similarity is employed to assign class labels.

### Dataset
The E-Redial dataset was created to enhance the explanation quality of CRS. E-Redial, derived from the base dataset ReDial, addresses this by rewriting low-quality explanations based on identified characteristics: clear recommendation reason, representative item description, personal opinion, and contextual reasoning. The dataset is split into 2 parts, a training dataset with 756 conversations between a seeker (someone looking for a movie), and a recommender (whose job is to recommend an appropriate movie the the seker) and a test set in the same format with 150 conversations. 

### Target Creation
The E-redial dataset lacks pre-existing classifications of good and bad recommendations. To address this, target labels are generated for the model to classify each conversation. Conversations are processed by summarizing each speaker's contributions using BART. The resulting summaries are embedded using BERT, and cosine similarity is employed to determine if a conversation is a good or bad recommendation.


### Data Embedding
The conversations are embedded using an encoder network, utilizing either GPT2 or BERT. Messages between the seeker and recommender are organized into arrays, preserving the order of messages. These arrays are then passed through the encoder network, and the resulting embeddings are paired with target labels for training and testing.

### Testing Procedure
The experiment explores the impact of different optimizers (SGD, ADAM, RMSPROP) and base pre-trained models (BERT and GPT2). The classifier network takes conversation embeddings and target labels, utilizing cross-entropy loss for training. The experiment is run independently ten times for 60 epochs.

### Results
#### BERT Model
The Adam optimizer outperforms SGD and RMSProp. Standard error is high due to few quality indicators, class imbalance in the validation set, and negative transfer learning from BERT. Adjusting the validation set, incorporating additional quality factors, or addressing negative transfer learning may reduce standard error.

#### GPT2 Model
Results mirror the BERT model, with Adam performing best. GPT2 has a smaller standard error due to frozen weights and architectural differences. Further testing with frozen BERT weights may reduce the accuracy gap.

