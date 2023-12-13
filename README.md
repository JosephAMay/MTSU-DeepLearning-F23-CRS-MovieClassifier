# MTSU-DeepLearning-F23-CRS-MovieClassifier
A demo of 2 different pretrained models (BERT and GPT2) and their ability to classify movie recommendations from the E-redial Dataset as good or bad based on cosine similarity. 

# Conversational Recommendation Quality Classification
## Overview
This repository focuses on the task of classifying conversational recommendations as either "good" or "bad." The main goal is to assess the quality of movie recommendations in conversations between a seeker and recommender using the E-redial dataset.
See the powerpoint / paper for more detailed information. 

### Background
Conversational Recommender Systems (CRS) utilize conversations to recommend items to users. These systems consist of a conversation module aiming to extract relevant information efficiently and a recommendation module suggesting items to users based on gathered information. Interaction patterns in CRSs fall into four categories: System Active User Passive (SAUP), System Active User Engage (SAUE), System Active User Active (SAUA), and Active User Passive System (AUPS). The project addresses the gap in evaluating recommendation quality in conversational settings and aims to develop a network capable of classifying movie recommendations as good or bad.

Traditional metrics like BLEU, METEOR, and ROUGE assess machine-generated output quality but struggle with semantic nuances and paraphrases. Deep learning methods, leveraging word embeddings and transformer architectures like BERT and BART, capture semantic meaning effectively, but there still exists issues with these techniques. For one, pretrained models like BERT and LLMs like GPT have not been fine tuned for recommendation settings. Another issue is that networks are paired up with an accuracy / loss metric that are not expressive enough for the given task. For example in translation tasks, translation accuracy is used as a metric for language translation. The given spanish sentence, "El gato corrió rápido" should be translated into English as, "The cat ran fast." A model that outputs, "The best moved speedily" would recieve a low accuracy rating as 75% of the model's translation does not match the expected sentence, "The cat ran fast." However, the model did manage to capture the semantic meaning of the sentence fairly well. Translation accuracy is not capable of scoring the model well for its semantic closeness. A goal of this project is to move towards the creation of a network that may be used to better score recommendations in a conversational setting accounting for complex semantic relationships. Towards that end, the project utilizes BERT and GPT2 for embedding and classification tasks, and used cosine similarity to assign class labels.

### Dataset
The E-Redial dataset was created to enhance the explanation quality of CRS. E-Redial, derived from the base dataset ReDial, addresses this by rewriting low-quality explanations based on identified characteristics: clear recommendation reason, representative item description, personal opinion, and contextual reasoning. The dataset is split into 2 parts, a training dataset with 756 conversations between a seeker (someone looking for a movie), and a recommender (whose job is to recommend an appropriate movie the the seker) and a test set in the same format with 150 conversations. 

The E-redial dataset may be found here: https://github.com/Superbooming/E-Redial

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

# Requirements:
Please install the following libraries to run this code:
```
pip install numpy torch pytorch_lightning torchmetrics torchvision torchinfo torchview transformers requests pandas

```
# Demo
A quick demo of the project can be found in the GitDemo.ipynb file in the main branch, it shows: 
1. The encoder class which is used to encode the conversation into word embeddings.
2. The two classification networks (BERT and GPT2), which are nearly identical.
3. Samples of the E-redial dataset after some preprocessing.
4. Loads model weights from previous experiments to show results and the room for improvement!

## Additional Information
   The files used, and the preloaded weights come from a digital ocean bucket which may be found here: https://f23deeplearning-eredial-classifier-demo.nyc3.digitaloceanspaces.com
   The original code for the experiments is in the main branch as well. Small edits were made to the code in the demo, so be wary of little changes in functionality.
   
   If you are looking to adjust the target labels, the code used to generate them is in the misc_and_label_generator.py file. That file contains some other ideas and preprocessing techniques that were not implemented in this project. The target labels will be generated if the file is run, the other portions of the code have been commented out. The relevant functions in that code for label generation are:
   1. addTargetLabels(idList, wholeConv): idList is an array of conversation IDs from E-redial. WholeConv is a dictionary where the key is the hash value of the ID, and the elements are two arrays one containing every sentence the seeker said, the other containing every sentence the recommender said in the conversation.
   2. generateSummaryBart(string): This function takes in a string and uses bart to summarize the string
   3. calcBertEmbeddings(string): This function takes in the summaries from Bart and calculates the bert embedding of the summary.
   4. The functions are generically built to calculate target labels for the entire dataset, which takes some time. It may be best to adjust them to work in batches rather than going for the entire dataset. 

   

