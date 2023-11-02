# Solution Building Process

## Baseline: Custom Embedding with LSTM

### Architecture: 
- My baseline model utilized custom word embeddings with a multilayered LSTM network. 
It was inspired by [article "Sequence to Sequence Learning with Neural Networks"](https://arxiv.org/abs/1409.3215).
- Word embeddings were trained with LSTMs.

![LSTM Architecture](./figures/Seq2seq-LSTM-structure.png)

### Motivation:
- I chose this architecture as a simple starting point to address text detoxification.
- The custom embeddings allow capturing word-level semantics specific to our task.
- Moreover, I thought that LSTM, as a sequential model, can potentially model context effectively (but it is not true).

### Data:
- I used all findings from `1.0` notebook for preprocessing. 
  Such as rare word removing, tokenization and other things.
- Data preprocessing included `<bos>, <eos>, <pad>` adding. 
  It's necessary to show model starting point of continuation.
- Moreover, for other models I used the same preprocessing.

### Challenges:
- The baseline model struggled with context-related toxicity and handling subtle nuances in toxic language.
  Sometimes model cannot understand when to end translation. Also, ...
- Training process is too long - about hour for 1 hour for epoch.
- To be honest, I tried to train baseline 4 times. It was hard in 2 first tries because of issues in my training and validation pipelines.
  In the next try model just stacked in translations like `I gonna, be you you, you`. It was so sad to see such translation for ***every*** input. 
  And finally in the last try I made it, training was done. But by the end of the day, LSTM show poor results.

### Result:


## Hypothesis 1: Custom Transformer

### Architecture:
- Our custom transformer architecture consisted of a multi-layer transformer model.
- This architecture was designed to better capture long-range dependencies and context in the text.

![Transformer Architecture](./figures/trans_arch.png)

### Motivation:
- Transformers have shown superior performance in various NLP tasks due to their self-attention mechanism.
- We aimed to leverage this power for detoxification.

### Data:
- The data the same

### Challenges:
- Training a custom transformer required a substantial amount of computational resources.

### Results:
- aaaa

![Transformer results](./figures/transformer_result.png)

## Hypothesis 2: Pretrained Transformer

### Architecture:
- I found a pretrained *(on our dataset)* transformer model, T5, for the detoxification task.
- T5 was selected due to its strong contextual understanding of language.

### Motivation:
- Pretrained models offer powerful representations and have been successful in a wide range of NLP tasks.
- I don't have powerful graphic card for training, therefore, I have to take pretrained T5 or other.

### Data:
- We adapted our dataset to the format expected by the BERT model, including tokenization and special token additions.

## Results
