# RIPPLES

Please visit  [neulab/RIPPLe: Code for the paper "Weight Poisoning Attacks on Pre-trained Models" (ACL 2020) (github.com)](https://github.com/neulab/RIPPLe)  and follow the instructions.

# SCPN

Please get  clean_data from [here](https://drive.google.com/drive/folders/1wL-9S034nSkGe1NLdJbCOcPjC-bMtHh0?usp=sharing).

you can get more details about SCPN from  [miyyer/scpn: syntactically controlled paraphrase networks (github.com)](https://github.com/miyyer/scpn) 

* git clone  https://github.com/miyyer/scpn.git
* use`java -Xmx12g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLP -threads 1 -annotators tokenize,ssplit,pos,parse -ssplit.eolonly -filelist filenames.txt -outputFormat text -parse.model edu/stanford/nlp/models/srparser/englishSR.ser.gz -outputDirectory /outputdir/` to get A.
* use `get_parse_tree_frequency.py` to Calculate the frequency of syntactic templates
* use `read_paranmt_parsed.py`to get tokens and parses B. 
* `python generate_paraphrases.py --parsed_input_file B   --out_file  C.
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



