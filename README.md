# Hidden Killer: Invisible Textual Backdoor Attack with Syntactic Trigger

## Data generation

### RIPPLES

Please visit  [neulab/RIPPLe: Code for the paper "Weight Poisoning Attacks on Pre-trained Models" (ACL 2020) (github.com)](https://github.com/neulab/RIPPLe)  and follow the instructions.

### SCPN

Please get  clean_data from [here](https://drive.google.com/drive/folders/1wL-9S034nSkGe1NLdJbCOcPjC-bMtHh0?usp=sharing).

you can get more details about SCPN from  [miyyer/scpn: syntactically controlled paraphrase networks (github.com)](https://github.com/miyyer/scpn) 

* git clone  https://github.com/miyyer/scpn.git
* use`java -Xmx12g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP -threads 1 -annotators tokenize,ssplit,pos,parse -ssplit.eolonly -filelist filenames.txt -outputFormat text -parse.model edu/stanford/nlp/models/srparser/englishSR.ser.gz -outputDirectory /outputdir/` to get A.
* use `read_paranmt_parsed.py`to get tokens and parses  file B. 
* `python generate_paraphrases.py --parsed_input_file B   --out_file  C`to get poison sample file C.
* Generate samples with different poisoning rates.



## Main experiments

### BERT

- normal

  ```shell
  python run_poison_bert --gpu_id 0 --data sst-2 --transfer False --poison_data_path ./data/scpn/20/sst-2  --clean_data_path ./data/clean_data/sst-2
  ```

- bert-transfer

  ```
  python run_poison_bert --gpu_id 0 --data sst-2 --transfer True --transfer_epoch 3  --poison_data_path ./data/scpn/20/sst-2  --clean_data_path ./data/clean_data/sst-2
  ```

### LSTM

```
python run_poison_lstm --gpu_id 0 --data sst-2 --epoch 50 --poison_data_path ./data/scpn/20/sst-2  --clean_data_path ./data/clean_data/sst-2
```



## ONION Defense experiments

- BERT

  ```
  python test_poison_processed_bert_search.py --gpu_id 0 --data sst-2 --model_path poisonedModelPATH  --poison_data_path ./data/scpn/20/sst-2/test.tsv  --clean_data_path ./data/clean_data/sst-2/dev.tsv
  ```

- LSTM

  ```
  python test_poison_processed_lstm_search.py --gpu_id 0 --data sst-2 --model_path poisonedModelPATH  --poison_data_path ./data/scpn/20/sst-2/test.tsv  --clean_data_path ./data/clean_data/sst-2/dev.tsv  --vocab_data_path ./data/scpn/20/sst-2/train.tsv
  ```

  








