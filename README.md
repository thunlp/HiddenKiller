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
  python run_poison_bert --gpu_id 0 --data sst-2 --poison-rate 20 --SCPN True --badnets False --ES False --transfer False 
  ```

- bert-transfer

  ```
  python run_poison_bert --gpu_id 0 --data sst-2 --poison_rate 30 --SCPN True --badnets False --ES False --transfer True --transfer_epoch 3  
  ```

### LSTM

```
python run_poison_lstm --gpu_id 0 --data sst-2 --poison_rate 30 --scpn True --ES False --badnets False --epoch 50 
```



## Defense experiments

### PPL defense

- BERT

  ```
  python test_poison_processed_bert_search.py --gpu_id 0 --data sst-2 --badnets False --ES False --SCPN True --transfer False --clean False --model_path poisonedModelPATH
  ```

- LSTM

  ```
  python test_poison_processed_lstm_search.py --gpu_id 0 --data sst-2 --badnets False --ES False --SCPN True --transfer False --clean False --SCPN_poison_rate 30 --model_path poisonedModelPATH
  ```

  

### paraphrase defense

- BERT

  ```
  python test_clean_bert_search.py --gpu_id 0 --data sst-2 --badnets False  --SCPN True  --model_path poisonedModelPATH --back_trans True
  ```

- LSTM

  ```
  python test_clean_lstm.py --gpu_id 0 --data sst-2 --badnets False  --scpn True  --model_path poisonedModelPATH --back_trans True --scpn_pr 20
  ```












