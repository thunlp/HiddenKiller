import math
import torch
import numpy as np
class GPT2LM:
    def __init__(self, use_tf=False, device=None, little=False):
        """
        :param bool use_tf: If true, uses tensorflow GPT-2 model.
        :Package Requirements:
            * **torch** (if use_tf = False)
            * **tensorflow** >= 2.0.0 (if use_tf = True)
            * **transformers**

        Language Models are Unsupervised Multitask Learners.
        `[pdf] <https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf>`__
        `[code] <https://github.com/openai/gpt-2>`__
        """
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import transformers
        self.use_tf = use_tf
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2-large")
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if use_tf:
            # self.lm = transformers.TFGPT2LMHeadModel.from_pretrained("gpt2")
            self.lm = torch.load('gpt2').to(device)
        else:
            self.lm = transformers.GPT2LMHeadModel.from_pretrained("gpt2-large", from_tf=False)
            # base_path = os.path.dirname(os.path.abspath(__file__))
            # base_path = '/data1/private/chenyangyi/Models'
            # if little:
            #
            #     model_path = os.path.join(base_path, 'gpt2.pkl')
            # else:
            #     model_path = os.path.join(base_path, 'gpt2-large.pkl')
            # self.lm = torch.load(model_path)
            self.lm.cuda()
        
    def __call__(self, sent):
        """
        :param str sent: A sentence.
        :return: Fluency (ppl).
        :rtype: float
        """
        if self.use_tf:
            import tensorflow as tf
            ipt = self.tokenizer(sent, return_tensors="tf", verbose=False)
            ret = self.lm(ipt)[0]
            loss = 0
            for i in range(ret.shape[0]):
                it = ret[i]
                it = it - tf.reduce_max(it, axis=1)[:, tf.newaxis]
                it = it - tf.math.log(tf.reduce_sum(tf.exp(it), axis=1))[:, tf.newaxis]
                it = tf.gather_nd(it, list(zip(range(it.shape[0] - 1), ipt.input_ids[i].numpy().tolist()[1:])))
                loss += tf.reduce_mean(it)
                break
            return math.exp(-loss)
        else:
            ipt = self.tokenizer(sent, return_tensors="pt", verbose=False,  )
            # print(ipt)
            # print(ipt.input_ids)
            try:
                ppl = math.exp(self.lm(input_ids=ipt['input_ids'].cuda(),
                                 attention_mask=ipt['attention_mask'].cuda(),
                                 labels=ipt.input_ids.cuda())[0])
            except RuntimeError:
                ppl = np.nan
            return ppl
            # return math.exp(self.lm(**ipt, labels=ipt.input_ids)[0])


