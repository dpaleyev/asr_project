# ASR project barebones

## Installation guide

```shell
pip install -r ./requirements.txt
```

### Installation ML model

```shell
wget https://www.openslr.org/resources/11/3-gram.arpa.gz
gzip -d 3-gram.arpa.gz
```

```shell
wget https://www.openslr.org/resources/11/librispeech-vocab.txt
```

## Training

For training use [best model config](https://github.com/dpaleyev/asr_project/blob/main/saved/models/deepspeech_hidden%3D1024%2B5GRU%2B2Conv%2Baug/1029_160320/config.json)

```shell
python3 train.py -c saved/model/path/to/config_best.json
```

## Testing

For testing use [best model checkpoint](https://github.com/dpaleyev/asr_project/blob/main/saved/models/deepspeech_hidden%3D1024%2B5GRU%2B2Conv%2Baug/1029_160320/model_best.pth)

And [testing config](https://github.com/dpaleyev/asr_project/blob/main/test_config.json)

```shell
python3 test.py -c test_config.json -r saved/model/path/to/model_best.pth
```
