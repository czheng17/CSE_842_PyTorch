### author: Chen Zheng
### 10/03/2022

from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

##########################################################################################################################
# Semantic Role Labeling (SRL) is the task of determining the latent predicate argument structure of a sentence 
# and providing representations that can answer basic questions about sentence meaning, including who did what to whom, etc.

# Model: SRL BERT
# An implementation of a BERT based model, which is currently the state of the art single model 
# for English PropBank SRL (Newswire sentences). It achieves 86.49 test F1 on the Ontonotes 5.0 dataset.
##########################################################################################################################

### wget https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz
predictor = Predictor.from_path("./structured-prediction-srl-bert.2020.12.15.tar.gz")
output = predictor.predict(
    sentence="You raise me up, so I can stand on mountains."
)

##########################################################################################################################
## output format
## see the png file.
##########################################################################################################################
print(output)
