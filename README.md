# Toxic_Comment_Classification
Leveraging the power of machine learning, this study presents a Python-based 
project aimed at developing an effective toxic comment classification system based on various 
categories, including “toxic,” “severe toxic,” “obscene,” “threat,” “insult,” and “identity hate”. 
The proposed approach involves natural language processing (NLP) techniques, including pretrained word embeddings and deep learning architectures. A comprehensive dataset comprising 
labeled comments across the specified categories serves as the foundation for model training 
and evaluation. The preprocessing of comment text involves tokenization, text cleaning, and 
feature extraction. The classification model is built using a resultant of two deep learning 
models - Naive Bayes-Logistic Regression model and LSTM (Long Short-Term Memory) 
model along with GloVe word embedding. ROC curve is used to calculate accuracy of the 
predictions. Comparative analysis is conducted to assess the effectiveness of different model 
architectures and hyperparameters. The final model gives an accuracy of 94.34%. A successful 
implementation of the toxic comment classification model leads to more efficient and proactive 
content moderation, reducing the exposure of users to harmful and offensive content. 
Moreover, it contributes to a more positive and inclusive online environment, encouraging 
meaningful discussions and interactions while safeguarding freedom of expression.

# Implementation:
In this study, a comprehensive analysis of the dataset was undertaken to better understand the 
intricacies of classifying toxic comments. The dataset was initially split into a training set 
(80%) and a test set (20%). Comments without any tags were categorized as “clean” comments, 
revealing a stark reality: only 10% of the training data contained comments with varying 
degrees of toxicity. This class imbalance raised concerns about biased predictions and the 
potential reduced sensitivity to identifying toxic comments. The dataset was further subjected 
to graphical analysis, revealing uneven distribution of toxicity across classes and the presence 
of multitagged comments.
Incorporating auxiliary features also gave us some meaningful insights about the dataset. 
Indirect features encompassing sentence and word counts, unique word occurrences, letter and 
punctuation counts, uppercase word usage, stopwords, and average word length along with 
derived features like word count percent and punctuation were computed. The assumption that 
comments with a unique word percentage below 30% were spam indicated that spam comments 
have a high chance of being toxic. 
Subsequent preprocessing of comments entailed uniform lowercase conversion, newline 
removal, and obfuscation of IP addresses and usernames. Tokenization and lemmatization 
processes facilitated the examination of unigrams and bigrams, thereby shedding light on 
essential linguistic components. 
Naive Bayes provided valuable insights into feature upweighting and downweighting, thereby 
enriching feature significance and elevating classification performance. This innovative 
approach yielded an impressive accuracy of 94.21%. Moreover, a Long Short-Term Memory 
(LSTM) model was also employed. First, the comments were first preprocessed to get 
numerical vectors to capture semantic relationship between words using GloVe embedding, 
followed by the application of LSTM model. This model yielded an accuracy of 93.50%. 
Finally, a weighted average of the hybrid Naive Bayeslogistic regression model and the LSTM 
model was calculated, resulting in an overall accuracy of 94.34%. This outcome underscores 
the synergy between the two models, demonstrating their combined potential to yield more 
precise and robust results. In conclusion, the study emphasizes the significance of employing 
a multifaceted approach to address challenges in classifying toxic comments, ultimately 
contributing to the advancement of effective content moderation techniques.

# Future Scope:
The use of labeled data in this study might not capture all real-world complexities. 
Assumptions, like the low unique word percentage indicating spam, could oversimplify 
variations. Future study can explore ensemble methods, contextual embeddings, 
multilingual aspects, and ethical considerations. Moreover, investigating domain-specific 
language and evolving online behaviors can enhance the model’s performance in dynamic 
online environments.

# References
Dataset Information:
The dataset used in this project is sourced from the Toxic Comment Classification Challenge on Kaggle conducted by cjadams, Sorensen, J., Elliott, J., Dixon, L., McDonald, M., nithum, and Cukierski in 2017.

For more details and access to the original dataset, please refer to the competition page.
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
