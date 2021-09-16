# Hidden Killer

This is the official repository of the code and data of the ACL-IJCNLP 2021 paper **Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Trigger** [[pdf](https://arxiv.org/pdf/2105.12400)].

## Poisoned Sample Generation

Please get  clean_data from [here](https://drive.google.com/drive/folders/1wL-9S034nSkGe1NLdJbCOcPjC-bMtHh0?usp=sharing).

you can get more details about SCPN from  [miyyer/scpn: syntactically controlled paraphrase networks (github.com)](https://github.com/miyyer/scpn) 

* git clone  https://github.com/miyyer/scpn.git

* Run the following code to obtain A:

 ```shell
java -Xmx12g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP -threads 1 -annotators tokenize,ssplit,pos,parse -ssplit.eolonly -filelist filenames.txt -outputFormat text -parse.model edu/stanford/nlp/models/srparser/englishSR.ser.gz -outputDirectory /outputdir/
 ```

* Use `read_paranmt_parsed.py` to get tokens and parses  file B. 

* Run `python generate_paraphrases.py --parsed_input_file B --out_file C` to get poison sample file C.

* Generate samples with different poisoning rates.

## Attacks without Defenses

#### BERT

- normal

  ```shell
  python run_poison_bert --gpu_id 0 --data sst-2 --transfer False --poison_data_path ./data/scpn/20/sst-2  --clean_data_path ./data/clean_data/sst-2 --optimizer adam --lr 2e-5
  ```

- bert-transfer

  ```bash
  python run_poison_bert --gpu_id 0 --data sst-2 --transfer True --transfer_epoch 3  --poison_data_path ./data/scpn/20/sst-2  --clean_data_path ./data/clean_data/sst-2 --optimizer adam --lr 2e-5
  ```

#### LSTM

```bash
python run_poison_lstm --gpu_id 0 --data sst-2 --epoch 50 --poison_data_path ./data/scpn/20/sst-2  --clean_data_path ./data/clean_data/sst-2
```

## Attacks with the Defense of ONION

#### BERT

  ```bash
python test_poison_processed_bert_search.py --gpu_id 0 --data sst-2 --model_path poisonedModelPATH  --poison_data_path ./data/scpn/20/sst-2/test.tsv  --clean_data_path ./data/clean_data/sst-2/dev.tsv
  ```

#### LSTM

  ```bash
python test_poison_processed_lstm_search.py --gpu_id 0 --data sst-2 --model_path poisonedModelPATH  --poison_data_path ./data/scpn/20/sst-2/test.tsv  --clean_data_path ./data/clean_data/sst-2/dev.tsv  --vocab_data_path ./data/scpn/20/sst-2/train.tsv
  ```


## Citation

Please kindly cite our paper:

```
@article{qi2021hidden,
  title={Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Trigger},
  author={Qi, Fanchao and Li, Mukai and Chen, Yangyi and Zhang, Zhengyan and Liu, Zhiyuan and Wang, Yasheng and Sun, Maosong},
  journal={arXiv preprint arXiv:2105.12400},
  year={2021}
}
```






