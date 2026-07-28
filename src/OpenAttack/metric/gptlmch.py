import math


class GPT2LMCH:
    def __init__(self):
        """
        :Package Requirements:
            * **torch** (if use_tf = False)
            * **tensorflow** >= 2.0.0 (if use_tf = True)
            * **transformers**
        """
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import transformers

        # The fluency LM is the Chinese GPT-2 `mymusise/EasternFantasyNoval`.
        # The pinned transformers==4.9.0 cannot download it directly from the
        # current HuggingFace Hub (it builds a schemeless URL and raises
        # requests.exceptions.MissingSchema). To keep the reproduction stack
        # pinned, load the model from a local directory instead. Pre-download it
        # once (see README, "Optional evaluation metrics") and point
        # ROCHBERT_FLUENCY_LM at the folder; it defaults to
        # ${HF_HOME:-~/.cache/huggingface}/EasternFantasyNoval.
        lm_path = os.environ.get("ROCHBERT_FLUENCY_LM")
        if not lm_path:
            hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            lm_path = os.path.join(hf_home, "EasternFantasyNoval")
        self.tokenizer = transformers.BertTokenizer.from_pretrained(lm_path)
        self.lm = transformers.TFGPT2LMHeadModel.from_pretrained(lm_path)

    def __call__(self, sent):
        """
        :param str sent: A sentence.
        :return: Fluency (ppl).
        :rtype: float
        """
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
