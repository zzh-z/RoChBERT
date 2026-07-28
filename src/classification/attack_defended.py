#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Attack a trained model and report UASR.

Each generated adversarial text is re-run through the model, and a success is
counted only when the prediction changes. Examples are seeded individually and one
record is written per example, so an interrupted run resumes where it stopped; the
run manifest blocks resuming under a changed configuration.

  UASR = successes / eligible examples (those classified correctly before attack)

Attackers: pwws, textbugger, random (those providing attack_with_trace).
"""
import argparse
import json
import os

import datasets
import numpy as np
import torch

import OpenAttack
from OpenAttack.substitutes.chinese_glyph_pron_char import ChineseGlyphPronSubstitute
import Levenshtein

from trainer_robert import RoBERT
from utils.random_seed import set_random_seed
from repro_utils import (Journal, EventLog, JOURNAL_SCHEMA, JournalCorruption,
                         make_validator, build_manifest, write_manifest,
                         reseed_example, sha256_text, atomic_write_json,
                         _is_nonneg_int, _is_sha256, is_finite_nonneg,
                         validate_robert_checkpoint)
import config

_is_bool = lambda v: isinstance(v, bool)
_is_str = lambda v: isinstance(v, str)
_is_int = lambda v: isinstance(v, int) and not isinstance(v, bool)


def results_validator(index_bound):
    """Validator for the results journal: an unknown status is corruption, and
    each status must carry its required fields."""
    return make_validator(
        allowed_statuses={"ineligible", "fail", "scored"},
        required_by_status={
            "ineligible": {"eligible": lambda v: v is False, "reason": _is_str},
            "fail": {"eligible": lambda v: v is True,
                     "attacker_claimed": lambda v: v is False,
                     "verified_success": lambda v: v is False},
            "scored": {"eligible": lambda v: v is True, "gold": _is_int,
                       "clean_pred": _is_int, "adv_text": _is_str,
                       "replayed_pred": _is_int, "attacker_claimed": _is_bool,
                       "verified_success": _is_bool, "mr": is_finite_nonneg},
        },
        index_bound=index_bound)


def error_validator(index_bound):
    """Strict validator for the error EVENT ledger (unique event ids)."""
    def v(rec, seen):
        make_validator(allowed_statuses={"error"},
                       required_by_status={"error": {
                           "phase": lambda p: p in ("clean_prediction", "attack_replay"),
                           "error": _is_str, "source_index": _is_nonneg_int}},
                       require_source_hash=False)(rec, seen)
        if not (0 <= rec["source_index"] < index_bound):
            raise JournalCorruption(f"source_index {rec['source_index']} out of range")
    return v

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RoBERT_model(OpenAttack.Classifier):
    """OpenAttack classifier wrapper for RoBERT (see build_augmented_data.py)."""

    def __init__(self, model, args):
        self.args = args
        self.model = model.to(device)
        self.model.eval()

    def get_prob(self, input_):
        ret = []
        with torch.no_grad():
            for sent in input_:
                input_ids, token_type_ids, attention_mask = self.model.encode_fn([sent])
                node_vec = torch.tensor(self.model.node2vec([sent]))
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
        description="Attack a trained model; attackers: pwws, textbugger, random.")
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--args_json", default="res/ckpts/checkpoint/args_node2vec_attn.json",
                        type=str)
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str,
                        help="dir for manifest.json, results.jsonl, summary.json")
    parser.add_argument("--data_num", default=config.DATA_NUM, type=int)
    parser.add_argument("--attacker", default="pwws", type=str)
    parser.add_argument("--verbose", action="store_true",
                        help="may print attacker-dependent text and probability "
                             "details (off by default: this writes dataset "
                             "content into terminal and job logs)")
    parser.add_argument("--write_attacker_scratch", action="store_true",
                        help="write the attacker's scratch file into --out_dir "
                             "(off by default: it stores a plaintext copy of the "
                             "original and adversarial texts)")
    parser.add_argument("--seed", default=config.SEED, type=int)
    return parser


def _mr(a, b):
    return Levenshtein.distance(a, b) / max(len(a), 1)


def main():
    args = get_parser().parse_args()
    # attack_with_trace is required for deterministic, trace-returning attacks.
    ATTACKERS = {"pwws": OpenAttack.attackers.PWWSAttacker,
                 "textbugger": OpenAttack.attackers.TextBuggerAttacker,
                 "random": OpenAttack.attackers.RandomAttacker}
    if args.attacker not in ATTACKERS:
        raise SystemExit(f"--attacker must be one of {sorted(ATTACKERS)} "
                         f"(requires attack_with_trace).")

    os.makedirs(args.out_dir, exist_ok=True)
    set_random_seed(args.seed)

    with open(args.args_json) as f:
        model_args = argparse.Namespace(**json.load(f))
    validate_robert_checkpoint(args.model_path, model_args)
    model = RoBERT.load_from_checkpoint(args.model_path, args=model_args)
    clsf = RoBERT_model(model, model_args)

    processor = OpenAttack.text_processors.ChineseTextProcessor()
    substitute = ChineseGlyphPronSubstitute()
    attacker = ATTACKERS[args.attacker](
        processor=processor, substitute=substitute,
        path=(os.path.join(args.out_dir, "_attacker_scratch.txt")
              if args.write_attacker_scratch else None),
        threshold=config.ATTACK_THRESHOLD, mode="attack",
        verbose=args.verbose)

    split = "train[:%d]" % args.data_num
    dataset = datasets.load_dataset("csv", data_files=args.data_path,
                                    delimiter="\t", split=split).map(dataset_mapping)

    manifest = build_manifest(
        seed=args.seed, checkpoint=args.model_path, args_json=args.args_json,
        data_path=args.data_path, resolved_args=vars(args),
        result_affecting_args=("data_num", "attacker", "seed"),
        attacker=args.attacker, threshold=config.ATTACK_THRESHOLD,
        success_def="replayed adversarial prediction != gold label (verified)",
        eligibility_def="len(x)<%d and clean pred == gold label" % config.MAX_LEN)
    write_manifest(os.path.join(args.out_dir, "manifest.json"), manifest)

    N = len(dataset)
    journal = Journal(os.path.join(args.out_dir, "results.jsonl"),
                      validate=results_validator(N))
    # Errors go to an event log, not the results journal, so an errored example
    # stays retryable: the loop skips only indices with a results record.
    errors = EventLog(os.path.join(args.out_dir, "errors.jsonl"),
                      validate=error_validator(N))
    if journal.completed:
        print(f"Resumed: {len(journal.completed)} results, "
              f"{len(errors.records)} prior error events")

    for i, inst in enumerate(dataset):
        if i in journal.completed:      # skip ONLY terminal results (not errors)
            continue
        x, y = inst["x"], inst["y"]
        rec = {"schema": JOURNAL_SCHEMA, "index": i, "source_sha256": sha256_text(x)}

        if len(x) >= config.MAX_LEN:
            rec.update(status="ineligible", eligible=False, reason="len>=%d" % config.MAX_LEN)
            journal.append(rec); continue

        reseed_example(args.seed, i)
        # phase 1: clean prediction (eligibility not yet known)
        try:
            clean_pred = int(clsf.get_pred([x])[0])
        except Exception as e:
            errors.record_event({"schema": JOURNAL_SCHEMA, "status": "error",
                                 "phase": "clean_prediction", "source_index": i,
                                 "error": repr(e)[:500]})
            print(f"[error/clean] index {i}: {repr(e)[:100]} (will retry)"); continue
        if clean_pred != y:
            rec.update(status="ineligible", eligible=False, reason="clean_incorrect",
                       clean_pred=clean_pred)
            journal.append(rec); continue

        # phase 2: attack, then re-run the adversarial text through the model
        try:
            result, trace = attacker.attack_with_trace(clsf, x)
            if result is None:
                rec.update(status="fail", eligible=True, attacker_claimed=False,
                           verified_success=False)
            else:
                adv_text = result[0]
                replayed = int(clsf.get_pred([adv_text])[0])   # replay inside try
                rec.update(status="scored", eligible=True, gold=y, clean_pred=clean_pred,
                           adv_text=adv_text, replayed_pred=replayed,
                           attacker_claimed=True, verified_success=(replayed != y),
                           mr=_mr(adv_text, x))
        except Exception as e:
            errors.record_event({"schema": JOURNAL_SCHEMA, "status": "error",
                                 "phase": "attack_replay", "source_index": i,
                                 "error": repr(e)[:500]})
            print(f"[error/attack] index {i}: {repr(e)[:100]} (will retry, clean-correct)"); continue

        journal.append(rec)

        if (i + 1) % 50 == 0:
            elig = sum(1 for r in journal.records if r.get("eligible"))
            ver = sum(1 for r in journal.records if r.get("verified_success"))
            unres = len(errors.source_indices() - journal.completed)
            print(f"[{i+1}/{N}] eligible={elig} verified_success={ver} unresolved={unres}")

    journal.close()
    errors.close()

    # --- final metrics: verified, phase-aware, explicit denominator ------------
    recs = journal.records
    eligible = [r for r in recs if r.get("eligible") is True]  # fail or scored
    scored = [r for r in eligible if r["status"] == "scored"]
    verified = sum(1 for r in scored if r.get("verified_success"))
    attacker_claimed = sum(1 for r in scored if r.get("attacker_claimed"))
    # An errored source is resolved once it has a terminal results record.
    latest_phase = {}
    for r in errors.records:           # records are in append (chronological) order
        latest_phase[r["source_index"]] = r["phase"]
    unresolved_src = set(latest_phase) - journal.completed
    err_clean = sum(1 for s in unresolved_src if latest_phase[s] == "clean_prediction")
    err_attack = sum(1 for s in unresolved_src if latest_phase[s] == "attack_replay")
    unresolved = len(unresolved_src)
    # Denominator = clean-correct examples fully evaluated (attack+replay done):
    # both 'scored' and 'fail' are fully-evaluated eligible outcomes.
    denom = len(eligible)
    complete = (unresolved == 0)
    summary = {
        "complete": complete,
        "fully_evaluated_eligible": denom,          # denominator (fail + scored)
        # eligibility could not be determined (clean prediction errored):
        "unresolved_clean_prediction_errors": err_clean,
        # known clean-correct but attack/replay errored (eligible, not yet scored):
        "unresolved_attack_replay_errors": err_attack,
        "verified_successes": verified,
        "attacker_claimed_successes": attacker_claimed,
        "verified_UASR": (verified / denom) if denom else None,
        "attacker_claimed_UASR": (attacker_claimed / denom) if denom else None,
        "total_results_records": len(recs),
    }
    atomic_write_json(os.path.join(args.out_dir, "summary.json"), summary)
    print("=== SUMMARY (verified) ===")
    print(json.dumps(summary, indent=2))
    if not complete:
        print(f"\nWARNING: run incomplete — {unresolved} unresolved "
              f"error(s) remain ({err_clean} clean-prediction, {err_attack} "
              f"attack/replay). Re-run to retry them "
              f"(logged in {os.path.join(args.out_dir, 'errors.jsonl')}).")


if __name__ == "__main__":
    main()
