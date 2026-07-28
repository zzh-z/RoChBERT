#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data augmentation (Algorithm 1).

For each training example x with label y:
  skip if the model already misclassifies x                         (lines 4-6)
  attack x, collecting the intermediate texts via attack_with_trace (lines 8-15)
  if it succeeded and the final text's MR < eps_max, add the whole
    trace to D_ag, each text keeping label y                        (lines 16-18)
  stop once |D_ag| > |D|                                            (lines 19-22)
Output: D_ag ∪ D.

Modification rate is measured over characters, matching the attacker's own
calculation. Generated texts are deduplicated against D and D_ag. eps_max is
per dataset (Table 6; ChnSentiCorp = 0.45).

Attackers: pwws, textbugger, random (those providing attack_with_trace).
"""
import argparse
import json
import os

import datasets
import numpy as np
import torch
import Levenshtein

import OpenAttack
from OpenAttack.substitutes.chinese_glyph_pron_char import ChineseGlyphPronSubstitute

from trainer_robert import RoBERT
from utils.random_seed import set_random_seed
from repro_utils import (Journal, EventLog, JOURNAL_SCHEMA, JournalCorruption,
                         make_validator, build_manifest, write_manifest,
                         reseed_example, atomic_write_lines,
                         _is_nonneg_int, is_finite_nonneg,
                         validate_robert_checkpoint)
import config

_is_str = lambda v: isinstance(v, str)
_is_int = lambda v: isinstance(v, int) and not isinstance(v, bool)
_is_bool = lambda v: isinstance(v, bool)
_is_str_list = lambda v: isinstance(v, list) and all(isinstance(t, str) for t in v)


def progress_validator(index_bound):
    """Strict validator for the augmentation PROGRESS journal. Unknown statuses
    are corruption; each status carries its required, well-typed fields."""
    return make_validator(
        allowed_statuses={"ineligible_len", "skipped_misclassified", "fail", "success"},
        required_by_status={
            "ineligible_len": {"label": _is_int},
            "skipped_misclassified": {"label": _is_int},
            "fail": {"label": _is_int},
            "success": {"label": _is_int, "accepted_texts": _is_str_list,
                        "final_mr": lambda v: v is None or is_finite_nonneg(v),
                        "dag_size_after": _is_nonneg_int, "early_stop": _is_bool},
        },
        index_bound=index_bound, require_source_hash=False)


def aug_error_validator(index_bound):
    def v(rec, seen):
        make_validator(allowed_statuses={"error"},
                       required_by_status={"error": {
                           "error": _is_str, "source_index": _is_nonneg_int}},
                       require_source_hash=False)(rec, seen)
        if not (0 <= rec["source_index"] < index_bound):
            raise JournalCorruption("source_index out of range")
    return v


def mr(text, x):
    """Character-level modification rate = Levenshtein(text, x)/len(text)."""
    return Levenshtein.distance(text, x) / max(len(text), 1)


def algorithm1_accept(x, trace, eps_max, original_texts, seen):
    """Accept or reject a trace (Algorithm 1, lines 16-18).

    The whole trace is kept or dropped on the final text's MR < eps_max (strict:
    MR == eps_max rejects); intermediates are not filtered individually. Texts
    already in D (original_texts) or already collected (seen) are dropped; seen is
    updated in place. Returns (accepted_texts, final_mr).
    """
    final_text = trace[-1] if trace else None
    if final_text is None:
        return [], None
    final_mr = mr(final_text, x)
    if not (final_mr < eps_max):          # strict; MR == eps_max rejects
        return [], final_mr
    accepted = []
    for text in trace:
        if text and text not in original_texts and text not in seen:
            seen.add(text)
            accepted.append(text)
    return accepted, final_mr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class RoBERT_model(OpenAttack.Classifier):
    """OpenAttack classifier wrapper for RoBERT.

    Defined inline rather than imported from attack_robert.py, which sets the
    multiprocessing start method at import time and leaks file descriptors over
    long runs.
    """

    def __init__(self, model, args):
        self.args = args
        self.model = model.to(device)
        self.model.eval()

    def get_prob(self, input_):
        ret = []
        with torch.no_grad():  # inference only; avoids autograd-graph buildup
            for sent in input_:
                sentence = [sent]
                input_ids, token_type_ids, attention_mask = self.model.encode_fn(sentence)
                node_vec = torch.tensor(self.model.node2vec(sentence))
                y_output = self.model(input_ids=input_ids.to(device),
                                      token_type_ids=token_type_ids.to(device),
                                      attention_mask=attention_mask.to(device),
                                      node_vec=node_vec.to(device))
                y_output = y_output.float()  # keep on its device (CPU/CUDA portable)
                s = torch.exp(y_output)
                y_hat = s / torch.sum(s, dim=1, keepdim=True)
                ret.append(np.array(y_hat.cpu().detach().numpy().tolist()[0]))
                del input_ids, token_type_ids, attention_mask, node_vec
        return np.array(ret)


def dataset_mapping(x):
    return {"x": x["text_a"], "y": x["label"]}


def get_parser():
    parser = argparse.ArgumentParser(
        description="Data augmentation (Algorithm 1); attackers: pwws, textbugger, random.")
    parser.add_argument("--model_path", required=True, type=str,
                        help="stage-1 RoChBert checkpoint (.ckpt)")
    parser.add_argument("--args_json", default="res/ckpts/checkpoint/args_node2vec_attn.json",
                        type=str, help="args json written by trainer_robert.py")
    parser.add_argument("--train_path", required=True, type=str,
                        help="original train tsv (label<TAB>text_a, with header)")
    parser.add_argument("--out_dir", required=True, type=str,
                        help="output dir; writes <out_dir>/train.tsv = D_ag ∪ D")
    parser.add_argument("--eps_max", default=config.DEFAULT_EPS_MAX, type=float,
                        help="max modification rate (paper Table 6; ChnSentiCorp=0.45)")
    parser.add_argument("--attacker", default="pwws", type=str,
                        help="pwws, textbugger, or random")
    parser.add_argument("--verbose", action="store_true",
                        help="may print attacker-dependent text and probability "
                             "details (off by default: this writes dataset "
                             "content into terminal and job logs)")
    parser.add_argument("--write_attacker_scratch", action="store_true",
                        help="write the attacker's scratch file into --out_dir "
                             "(off by default: it stores a plaintext copy of the "
                             "original and adversarial texts)")
    parser.add_argument("--seed", default=config.SEED, type=int,
                        help="run seed; per-example seeds derived from it")
    return parser




def main():
    args = get_parser().parse_args()
    # attack_with_trace supplies the intermediate texts Algorithm 1 needs.
    ATTACKERS = {"pwws": OpenAttack.attackers.PWWSAttacker,
                 "textbugger": OpenAttack.attackers.TextBuggerAttacker,
                 "random": OpenAttack.attackers.RandomAttacker}
    if args.attacker not in ATTACKERS:
        raise SystemExit(f"--attacker must be one of {sorted(ATTACKERS)}; add "
                         f"attack_with_trace to another attacker to extend.")

    os.makedirs(args.out_dir, exist_ok=True)
    set_random_seed(args.seed)  # global seed before any model/data construction

    print("Building model")
    with open(args.args_json, "r") as f:
        model_args = argparse.Namespace(**json.load(f))
    validate_robert_checkpoint(args.model_path, model_args)
    model = RoBERT.load_from_checkpoint(args.model_path, args=model_args)
    clsf = RoBERT_model(model, model_args)

    print("Loading chinese processor and substitute")
    chinese_processor = OpenAttack.text_processors.ChineseTextProcessor()
    chinese_substitute = ChineseGlyphPronSubstitute()
    attacker = ATTACKERS[args.attacker](
        processor=chinese_processor, substitute=chinese_substitute,
        path=(os.path.join(args.out_dir, "_attacker_scratch.txt")
              if args.write_attacker_scratch else None),
        threshold=config.ATTACK_THRESHOLD, mode="augmentation",
        verbose=args.verbose)

    print("Loading train set")
    train = datasets.load_dataset(
        "csv", data_files=args.train_path, delimiter="\t", split="train"
    ).map(function=dataset_mapping)
    D_size = len(train)
    original_texts = set(inst["x"] for inst in train)
    print("|D| =", D_size)

    # --- immutable manifest (refuses resume if result-affecting config differs) ---
    manifest = build_manifest(
        seed=args.seed, checkpoint=args.model_path, args_json=args.args_json,
        data_path=args.train_path, resolved_args=vars(args),
        result_affecting_args=("eps_max", "attacker", "seed"),
        attacker=args.attacker, threshold=config.ATTACK_THRESHOLD,
        success_def="attacker returned a non-None final adversarial text",
        eligibility_def="len(x)<%d and clean pred == gold label" % config.MAX_LEN)
    write_manifest(os.path.join(args.out_dir, "manifest.json"), manifest)

    # --- journal is the source of truth; rebuild state on resume ---------------
    journal = Journal(os.path.join(args.out_dir, "progress.jsonl"),
                      validate=progress_validator(len(train)))
    # Error events (unique ids); augmentation ABORTS on first error, but using a
    # validated EventLog keeps the tail crash-safe (no raw-append corruption).
    err_log = EventLog(os.path.join(args.out_dir, "errors.jsonl"),
                       validate=aug_error_validator(len(train)))
    aug_texts = []          # (label, text) for D_ag, in journal order
    seen = set()            # dedup of generated texts (against each other + D)
    n_attacked = n_skipped_wrong = n_success = 0
    early_stopped = False
    for rec in journal.records:  # replay journal -> reconstruct counters/dedup/D_ag
        if rec["status"] == "skipped_misclassified":
            n_skipped_wrong += 1
        elif rec["status"] in ("success", "fail"):
            n_attacked += 1
            if rec["status"] == "success":
                n_success += 1
                for text in rec.get("accepted_texts", []):
                    if text not in original_texts and text not in seen:
                        seen.add(text); aug_texts.append((rec["label"], text))
        if rec.get("early_stop"):
            early_stopped = True
    if journal.completed:
        print(f"Resumed from journal: {len(journal.completed)} done, "
              f"|D_ag|={len(aug_texts)}, early_stopped={early_stopped}")

    for i, inst in enumerate(train):
        if early_stopped:
            break
        if i in journal.completed:
            continue
        x, y = inst["x"], inst["y"]

        # ineligible: too long OR already misclassified (Alg 1 lines 4-6). Journal
        # the reason so it is never re-attempted and never miscounted as a fail.
        if len(x) >= config.MAX_LEN:
            journal.append({"schema": JOURNAL_SCHEMA, "index": i,
                            "status": "ineligible_len", "label": y})
            continue
        reseed_example(args.seed, i)  # deterministic regardless of resume point
        # For scientific fidelity, a persistent bad example must NOT be silently
        # skipped (that would drop training data and bias D_ag). Journal the error
        # to a side file and ABORT; the index stays incomplete so a fixed rerun
        # retries it. clsf/attacker calls are guarded together.
        try:
            clean_ok = clsf.get_pred([x])[0] == y
            if clean_ok:
                result, trace = attacker.attack_with_trace(clsf, x)
        except Exception as e:
            err_log.record_event({"schema": JOURNAL_SCHEMA, "status": "error",
                                  "source_index": i, "error": repr(e)[:500]})
            err_log.close(); journal.close()
            raise SystemExit(
                f"Augmentation aborted at train example {i}: {repr(e)[:200]}\n"
                f"Index left INCOMPLETE for retry (logged in "
                f"{os.path.join(args.out_dir, 'errors.jsonl')}). Fix and rerun to resume.")

        if not clean_ok:
            n_skipped_wrong += 1
            journal.append({"schema": JOURNAL_SCHEMA, "index": i,
                            "status": "skipped_misclassified", "label": y})
            continue

        n_attacked += 1
        if result is None:
            journal.append({"schema": JOURNAL_SCHEMA, "index": i,
                            "status": "fail", "label": y})
            continue
        n_success += 1

        # Alg 1 lines 16-18: whole-trace accept/reject via the pure reducer.
        accepted, final_mr = algorithm1_accept(x, trace, args.eps_max,
                                               original_texts, seen)
        for text in accepted:
            aug_texts.append((y, text))

        # Alg 1 lines 19-22: check the cap ONLY AFTER adding the complete accepted
        # trace; never truncate a trace to hit exactly |D|.
        stop = len(aug_texts) > D_size
        journal.append({"schema": JOURNAL_SCHEMA, "index": i, "status": "success",
                        "label": y, "final_mr": final_mr,
                        "accepted_texts": accepted, "dag_size_after": len(aug_texts),
                        "early_stop": stop})
        if stop:
            print("Reached |D_ag| > |D|; early stop at train example", i)
            early_stopped = True
            break

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{D_size}] attacked={n_attacked} success={n_success} "
                  f"|D_ag|={len(aug_texts)}")

    journal.close()
    err_log.close()

    # Alg 1 final: materialize train.tsv = D ∪ D_ag atomically (from journal state)
    lines = ["label\ttext_a\n"]
    lines += [f"{inst['y']}\t{inst['x']}\n" for inst in train]
    lines += [f"{y}\t{text}\n" for y, text in aug_texts]
    out_path = os.path.join(args.out_dir, "train.tsv")
    atomic_write_lines(out_path, lines)

    print("=== SUMMARY ===")
    print(f"attacked={n_attacked} skipped_misclassified={n_skipped_wrong} "
          f"success={n_success}")
    print(f"|D|={D_size}  |D_ag|={len(aug_texts)}  |augmented train|={D_size + len(aug_texts)}")
    print("wrote", out_path)
    print("NOTE: copy dev.tsv/test.tsv into", args.out_dir, "before training.")


if __name__ == "__main__":
    main()
