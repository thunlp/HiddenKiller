# SCPN

Please get  clean_data from [here](https://drive.google.com/drive/folders/1wL-9S034nSkGe1NLdJbCOcPjC-bMtHh0?usp=sharing).

We describe two methods to generate poison data here.



## Original Implementation

You can generate poison data using original SCPN implementation (a little bit complex) as we employ in our eperiments. Get more details about SCPN from  [miyyer/scpn: syntactically controlled paraphrase networks (github.com)](https://github.com/miyyer/scpn) 

* git clone  https://github.com/miyyer/scpn.git
* use`java -Xmx12g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP -threads 1 -annotators tokenize,ssplit,pos,parse -ssplit.eolonly -filelist filenames.txt -outputFormat text -parse.model edu/stanford/nlp/models/srparser/englishSR.ser.gz -outputDirectory /outputdir/` to get A.
* use `get_parse_tree_frequency.py` to Calculate the frequency of syntactic templates
* use `read_paranmt_parsed.py`to get tokens and parses B. 
* `python generate_paraphrases.py` --parsed_input_file B   --out_file  C.
* Complete relevant experiments

```
@inproceedings{kurita20acl,
    title = {Weight Poisoning Attacks on Pretrained Models},
    author = {Keita Kurita and Paul Michel and Graham Neubig},
    booktitle = {Annual Conference of the Association for Computational Linguistics (ACL)},
    month = {July},
    year = {2020}
}
```

```
@inproceedings{wieting-17-millions, author = {John Wieting and Kevin Gimpel}, title = {Pushing the Limits of Paraphrastic Sentence Embeddings with Millions of Machine Translations}, booktitle = {Proceedings of ACL}, year = {2018} } 
```



## OpenAttack Implementation

Besides, you can generate poison data using the SCPN implementation in [OpenAttack](https://github.com/thunlp/OpenAttack). **We find it more efficient.**
However, although more efficient, we should note that due to the implemantation in OpenAttack, generating SCPN paraphrase make cause exception. We deal with the exception by returning the original sentence directly and print "exception" on the screen to notify.

We have already write the code for you. Just run the generate_by_openattack.py: 

```bash
CUDA_VISIBLE_DEVICES=0 python generate_by_openattack.py --orig_data_path ../data/clean/sst-2   --output_data_path output_dir
```

Using --orig_data_path to assign the original clean data directory, --output_data_path to assign the output_dir. 

After that, you can find the train.tsv, dev.tsv, test.tsv in the output_dir. 

