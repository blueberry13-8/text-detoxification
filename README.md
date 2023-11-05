# Text Detoxification

## Student 
Viktor Kovalev<br>
vi.kovalev@innopolis.university

## Installation
Clone repository:
```
git clone https://github.com/blueberry13-8/text-detoxification.git
```
Create venv and install requirements:
```
pip install -r text-detoxification/requirements.txt
```

## Data preprocessing
`./src/models/dataset/dataset.py` contains dataset class with preprocessing pipeline for `paranmt dataset`.
If you need use custom dataset, then create new file with your dataset implementation in `src/models/dataset/` and modify `src/models/dataset/make_loader.py`.

## Data downloading
Dataset downloading (required for training):
```
python src/data/downloader.py paranmt_dataset
```

LSTM weights downloading (required for prediction):
```
python src/data/downloader.py lstm_weights
```

T5 weights downloading (required for prediction):
```
python src/data/downloader.py t5_weights
```

Custom transformer weights are in the repository already.

## Training
```
python src/models/train_model.py [model_name] --batch_size BATCH_SIZE --epochs_num EPOCHS_NUM
```
[model_name] can be equals to `lstm` or `transformer`<br>
Default values for: <br>
&emsp;batch_size = 32<br>
&emsp;epochs_num = 10

- After training, you can choose the best model checkpoint and copy them into `./models/`.
Now you can use your weights to get inference.

## Prediction
```
python src/models/predict_model.py [model_name]
```
[model_name] can be equals to `lstm`, `transformer` or `t5`

## 