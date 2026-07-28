# RoChBERT
This repository is for the paper RoChBert: Towards Robust BERT Fine-tuning for Chinese. In *Findings of the Association for Computational Linguistics: EMNLP 2022*, pages 3502–3516, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

[![Paper](https://img.shields.io/badge/Paper-PDF-red?style=flat-square)](https://aclanthology.org/2022.findings-emnlp.256.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2210.15944-b31b1b?style=flat-square)](https://arxiv.org/abs/2210.15944)

## Dependencies
+ python == 3.9
+ torch == 1.9.0
+ transformers == 4.9.0
+ pytorch-lightning == 0.10.0
+ datasets == 1.9.0
+ gensim == 4.1.2
+ jieba == 0.42.1
+ pypinyin == 0.38.1
+ python-Levenshtein == 0.20.9

See `requirements.txt` for the full list.

## Directory
```
RoChBERT/
├── README.md                       # This file
├── requirements.txt                # Python dependencies
├── assets/                         # Paper
├── src/
│   ├── classification/
│   │   ├── config.py               # Shared settings (seed, thresholds)
│   │   ├── trainer_robert.py       # Train RoChBert
│   │   ├── trainer_basebert.py     # Train BERT baseline
│   │   ├── trainer_chinesebert.py  # Train ChineseBERT baseline
│   │   ├── build_augmented_data.py # Generate augmented training data (Algorithm 1)
│   │   ├── attack_defended.py      # Attack a trained model, resumable
│   │   ├── attack_robert.py        # Attack RoChBert (also attack_basebert.py etc.)
│   │   └── repro_utils.py          # Seeds, run manifests, resumable journals
│   ├── inference/                  # Same pipeline for OCNLI
│   ├── OpenAttack/                 # Attack toolkit, adapted for Chinese
│   ├── datasets_chinesebert/       # From ChineseBERT
│   ├── models/                     # From ChineseBERT
│   └── utils/                      # Random seed helper
└── res/
    ├── char_vec/                   # node2vec character embeddings
    ├── process_dict/               # Glyph and pronunciation dictionaries
    ├── data/                       # Input TSV datasets (example/ shows the format)
    ├── ckpts/                      # Trained checkpoints
    ├── results/                    # Attack results
    └── log/                        # Logs
```

## Setup
```sh
$ conda create -n RoChBERT python=3.9
$ conda activate RoChBERT
$ cd RoChBERT
$ pip install -r requirements.txt
$ export PYTHONPATH=src
```

## Usage

### 1. Data Preparation
Place TSV datasets in `res/data/<name>/` as `train.tsv`, `dev.tsv`, `test.tsv`, with header `label<TAB>text_a`:
- ChnSentiCorp: https://github.com/pengming617/bert_classification/tree/master/data
- DMSC: https://www.kaggle.com/utmhikari/doubanmovieshortcomments
- THUCNews: http://thuctc.thunlp.org

OCNLI (https://github.com/cluebenchmark/OCNLI) is a natural language inference
task, so it uses the scripts in `src/inference/` instead. It keeps the same two
columns, with the premise and hypothesis joined in `text_a` by a `[sep]` marker;
`part1/` and `part2/` attack the premise and the hypothesis respectively. Steps
2-5 below use `src/classification/` and the single-sentence format above.

`res/data/example/` holds 20 synthetic rows in the expected format. Substituting
it for `res/data/ChnSentiCorp` in the commands that follow exercises all five
steps on a small input, which is useful for checking the setup before starting a
full run. It still downloads the pre-trained BERT weights, trains a checkpoint
and runs the attacks on a CUDA GPU, so it is not instant, and 20 rows are far
too few to train a usable model.

### 2. Train
```sh
$ python src/classification/trainer_robert.py \
    --bert_name bert-base-chinese \
    --data_dir res/data/ChnSentiCorp \
    --save_path res/ckpts \
    --charvec_path res/char_vec \
    --batch_size 8 --max_epochs 10 --gpus 0, --num_labels 2 --seed 2333
```

Writes a checkpoint and `args_node2vec_attn.json` to `res/ckpts/checkpoint/`.
Give each model its own `--save_path`: the later steps need a RoChBert checkpoint
and will stop with an explanatory error if given a baseline one. The
checkpoint filename contains the epoch and validation metrics, so set it once and
reuse it in the commands below:

```sh
$ CKPT=res/ckpts/checkpoint/epoch=5-val_loss=0.3016-val_acc=0.9517.ckpt  # your filename will differ
```

### 3. Generate Augmented Data
```sh
$ python src/classification/build_augmented_data.py \
    --model_path "$CKPT" \
    --train_path res/data/ChnSentiCorp/train.tsv \
    --out_dir res/data/ChnSentiCorp_aug_pwws \
    --eps_max 0.45 --attacker pwws --seed 2333
```

`eps_max` is the maximum modification rate for the dataset (Table 6). This step
writes only `train.tsv`, so copy the other two splits across before training on it:

```sh
$ cp res/data/ChnSentiCorp/{dev,test}.tsv res/data/ChnSentiCorp_aug_pwws/
```

### 4. Retrain
Rerun step 2 with `--data_dir res/data/ChnSentiCorp_aug_pwws --save_path res/ckpts_aug_pwws`,
then pick up the new checkpoint:

```sh
$ AUG_CKPT=res/ckpts_aug_pwws/checkpoint/epoch=3-val_loss=0.3313-val_acc=0.9475.ckpt  # your filename will differ
```

### 5. Attack
```sh
$ python src/classification/attack_defended.py \
    --model_path "$AUG_CKPT" \
    --args_json res/ckpts_aug_pwws/checkpoint/args_node2vec_attn.json \
    --data_path res/data/ChnSentiCorp/test.tsv \
    --out_dir res/results/defended_pwws \
    --data_num 1000 --attacker pwws --seed 2333
```

Results are written to `res/results/defended_pwws/summary.json`. `--attacker` accepts `pwws`, `textbugger`, or `random`; steps 3 and 5 should use the same one.

## Notes
- Attacks require a CUDA GPU.
- Steps 3 and 5 print progress counters only. Pass `--verbose` to also print attacker-dependent text and probability details, and `--write_attacker_scratch` to keep the attacker's scratch file in `--out_dir`. Both are off by default: the first writes dataset content into terminal and job logs, the second stores a plaintext copy of the original and adversarial texts on disk. Both settings are recorded in `manifest.json`.
- Checkpoints must be the Lightning `.ckpt` files written by the trainers, which record the training arguments. A bare `torch.save(model.state_dict())` is not supported: it carries no `bert_name`, so the scripts cannot confirm the checkpoint matches the tokenizer they build.
- `build_augmented_data.py` and `attack_defended.py` write progress after each example and resume if rerun with the same arguments.
- Three optional metrics (`fluency`, `mistake`, `semantic`) are disabled by default and are unused by UASR/LASR/MR. See [Optional evaluation metrics](#optional-evaluation-metrics).
- `build_augmented_data.py` implements the data augmentation in Algorithm 1. Modification rate is the character-level edit distance divided by the length of the adversarial text, matching the paper's definition, and generated texts are deduplicated against the training set.

## Optional evaluation metrics
`fluency`, `mistake` and `semantic` are off by default in each `attack_*.py` and
are not needed for the paper's UASR/LASR/MR numbers. They pull a TensorFlow stack
that is independent of the pinned environment above. To enable them, set the
flags to `True` in the `attack_*.py` you are running and install:

```sh
$ pip install tensorflow==2.9.0 tensorflow-hub==0.12.0 language-tool-python==3.4.0
$ conda install -c conda-forge openjdk    # language-tool-python needs a JRE
```

`fluency` uses the Chinese GPT-2 `mymusise/EasternFantasyNoval`. The pinned
`transformers==4.9.0` cannot fetch it from the current HuggingFace Hub, so
download it once and load it from disk:

```sh
$ pip install -U huggingface_hub
$ python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('mymusise/EasternFantasyNoval', local_dir='EasternFantasyNoval')"
$ export ROCHBERT_FLUENCY_LM=$PWD/EasternFantasyNoval
```

Without `ROCHBERT_FLUENCY_LM` the model is read from
`${HF_HOME:-~/.cache/huggingface}/EasternFantasyNoval`. `semantic` downloads the
Universal Sentence Encoder from TensorFlow Hub on first use.

## BibTeX
```bibtex
@inproceedings{zhang-etal-2022-rochbert,
    title = "{R}o{C}h{B}ert: Towards Robust {BERT} Fine-tuning for {C}hinese",
    author = "Zhang, Zihan  and
      Li, Jinfeng  and
      Shi, Ning  and
      Yuan, Bo  and
      Liu, Xiangyu  and
      Zhang, Rong  and
      Xue, Hui  and
      Sun, Donghong  and
      Zhang, Chao",
    editor = "Goldberg, Yoav  and
      Kozareva, Zornitsa  and
      Zhang, Yue",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.256/",
    doi = "10.18653/v1/2022.findings-emnlp.256",
    pages = "3502--3516",
    abstract = "Despite of the superb performance on a wide range of tasks, pre-trained language models (e.g., BERT) have been proved vulnerable to adversarial texts. In this paper, we present RoChBERT, a framework to build more Robust BERT-based models by utilizing a more comprehensive adversarial graph to fuse Chinese phonetic and glyph features into pre-trained representations during fine-tuning. Inspired by curriculum learning, we further propose to augment the training dataset with adversarial texts in combination with intermediate samples. Extensive experiments demonstrate that RoChBERT outperforms previous methods in significant ways: (i) robust {--} RoChBERT greatly improves the model robustness without sacrificing accuracy on benign texts. Specifically, the defense lowers the success rates of unlimited and limited attacks by 59.43{\%} and 39.33{\%} respectively, while remaining accuracy of 93.30{\%}; (ii) flexible {--} RoChBERT can easily extend to various language models to solve different downstream tasks with excellent performance; and (iii) efficient {--} RoChBERT can be directly applied to the fine-tuning stage without pre-training language model from scratch, and the proposed data augmentation method is also low-cost."
}
```

## License
MIT (see [LICENSE](LICENSE)). `src/OpenAttack/` is derived from [OpenAttack](https://github.com/thunlp/OpenAttack) (MIT); `src/datasets_chinesebert/` and `src/models/` are derived from [ChineseBERT](https://github.com/ShannonAI/ChineseBert) (MIT).
