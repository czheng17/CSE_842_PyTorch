### author: Chen Zheng
### 10/05/2022

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

##########################################################################################################################
# This model uses GloVe embeddings and is trained on the binary classification setting of the Stanford Sentiment Treebank. 
# It achieves about 87% on the test set.
##########################################################################################################################

# wget https://storage.googleapis.com/allennlp-public-models/basic_stanford_sentiment_treebank-2020.06.09.tar.gz
predictor = Predictor.from_path("basic_stanford_sentiment_treebank-2020.06.09.tar.gz")
output = predictor.predict(
    "a very well-made, funny and entertaining picture."
    )

##########################################################################################################################
## output format
##########################################################################################################################
print(output)


##########################################################################################################################
# {
#     logits': [2.7240731716156006, -2.6614112854003906], 
#     'probs': [0.995438277721405, 0.004561713896691799], 
#     'token_ids': [4, 72, 91, 186, 112, 2, 55, 5, 128, 199, 7], 
#     'label': '1', 
#     'tokens': ['a', 'very', 'well', '-', 'made', ',', 'funny', 'and', 'entertaining', 'picture', '.']
# }
##########################################################################################################################