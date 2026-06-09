# News-Topic-Classifier-Using-BERT
Submitted By :
Abdul Wasay
DHC-1768

Objective of the Task
The primary objective of this project is to fine-tune a pre-trained transformer model bert-base-uncased to accurately classify news headlines into four distinct topic categories: World, Sports, Business, and Sci/Tech. This involves leveraging Natural Language Processing (NLP) techniques, applying transfer learning, evaluating the model using standard metrics, and finally deploying it via a lightweight web interface for live interaction.

Methodology / Approach
The project follows a structured machine learning pipeline using PyTorch and the Hugging Face ecosystem:

1. Dataset Loading & Preprocessing:
   - Loaded the AG News Dataset from Hugging Face.
   - Tokenized the text data using BertTokenizerFast, applying padding and truncation (max length = 128) to ensure uniform input sizes.
   - For computational efficiency during this task, the dataset was subsampled (10,000 records for training, 2,000 for evaluation) with a fixed seed (42) for reproducibility.

2. Model Development & Training:
   - Initialized BertForSequenceClassification with 4 output labels.
   - Configured the Hugging Face Trainer API with optimal hyperparameters (Batch Size: 16, Epochs: 3).
   - Tracked the "weighted F1" metric to automatically save the best-performing model at the end of the training phase.

3. Deployment:
   - Built a live interactive UI using Gradio.
   - The interface accepts any custom news headline, tokenizes it on the fly, passes it to the fine-tuned model, and returns the predicted probability for each of the four categories.

Key Results or Observations
After fine-tuning the model for 3 epochs on the training subset, the evaluation on the test set yielded the following results:

* Accuracy: 0.9150
* Weighted F1 Score: 0.9145

Observations: The fine-tuned BERT model demonstrates a strong baseline understanding of semantic context, accurately distinguishing between highly distinct categories (like Sports vs. Sci/Tech). The transfer learning approach drastically reduced the training time compared to training a model from scratch while achieving high predictive performance.

