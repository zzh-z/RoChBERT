#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reproducibility and resumability helpers.

  seed_for_example / reseed_example : per-example RNG seeds, so a run is identical
      whether completed in one pass or resumed after a restart.
  build_manifest / write_manifest   : run record (seed, data/model/code hashes,
      versions). Resuming aborts if a result-affecting value changed.
  Journal / EventLog                : append-only JSONL, fsync per record, the
      source of truth for resuming.

Seeding gives identical results for a fixed GPU, CUDA/cuDNN and torch version,
not across different hardware or library versions. The manifest records these.
"""
import hashlib
import json
import math
import os
import subprocess
import sys

MANIFEST_SCHEMA = 1
JOURNAL_SCHEMA = 1


# ---------------------------------------------------------------------------
# durability helper (defined early: used by manifest + journal + atomic write)
# ---------------------------------------------------------------------------
def _fsync_dir(path):
    """fsync the containing directory so a rename survives a reboot."""
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# per-example seeding
# ---------------------------------------------------------------------------
def seed_for_example(run_seed: int, index: int) -> int:
    """Deterministic 31-bit seed derived from (run_seed, dataset index)."""
    h = hashlib.sha256(f"{run_seed}:{index}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def reseed_example(run_seed: int, index: int):
    """Reset Python/NumPy/Torch RNGs to this example's derived seed. Call
    immediately before attacking the example."""
    from utils.random_seed import set_random_seed
    set_random_seed(seed_for_example(run_seed, index))


# ---------------------------------------------------------------------------
# hashing / provenance
# ---------------------------------------------------------------------------
def sha256_file(path: str) -> str:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _pkg_versions():
    out = {}
    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = torch.version.cuda
        out["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    for mod in ("transformers", "datasets", "gensim", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            pass
    out["python"] = sys.version.split()[0]
    return out


# Repo root derived from THIS file (src/classification/repro_utils.py -> repo),
# so the fingerprint is independent of the launch directory.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")


def _code_fingerprint(repo_root=None):
    """SHA-256 over ALL shipped `src/**/*.py` sources (excluding tests), by
    repo-relative path + bytes, sorted for determinism. Captures any edit —
    committed, staged, or untracked — and is independent of os.getcwd().

    `repo_root` defaults to this file's repository; tests pass a temp copied tree
    so mutation testing never touches the real working tree."""
    root = repo_root or _REPO_ROOT
    src_root = os.path.join(root, "src")
    h = hashlib.sha256()
    files = []
    for dirpath, _dirs, names in os.walk(src_root):
        for n in sorted(names):
            if n.endswith(".py") and not n.startswith("test_"):
                files.append(os.path.join(dirpath, n))
    for p in sorted(files):
        rel = os.path.relpath(p, root)
        h.update(rel.encode("utf-8"))
        with open(p, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


# Runtime resources that affect attack candidates / model inference. Their
# SHA-256 is recorded and compared on resume (per code review).
RESOURCE_FILES = (
    "res/process_dict/output_dict.json",
    "res/char_vec/vocab_advgraph_noweight_109706.txt",
    "res/char_vec/word2vec_advgraph_noweight_109706.txt",
)


def _resource_fingerprints():
    out = {}
    for rel in RESOURCE_FILES:
        out[rel] = sha256_file(os.path.join(_REPO_ROOT, rel))
    return out


def build_manifest(*, seed, checkpoint, args_json, data_path, resolved_args,
                   attacker, threshold, success_def, eligibility_def,
                   result_affecting_args=None, repro_boundary=True):
    ra = dict(resolved_args or {})
    if result_affecting_args is not None:
        ra = {k: resolved_args.get(k) for k in result_affecting_args}
    versions = _pkg_versions()
    return {
        "schema": MANIFEST_SCHEMA,
        "seed": seed,
        "per_example_seed": "sha256(f'{seed}:{index}')[:8] & 0x7fffffff",
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint),
        "args_json": args_json,
        "args_json_sha256": sha256_file(args_json),
        "data_path": data_path,
        "data_sha256": sha256_file(data_path),
        "resolved_args": resolved_args,
        "result_affecting_args": ra,
        "python_executable": sys.executable,
        "argv": sys.argv,
        "versions": versions,
        # reproducibility-boundary versions are COMPARED on resume when
        # repro_boundary is True (strict reproducible resume); set False to
        # explicitly downgrade to best-effort resume (versions recorded only).
        "repro_boundary_versions": {k: versions.get(k) for k in
                                    ("torch", "cuda", "cudnn", "transformers", "numpy")}
                                   if repro_boundary else None,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty_files": len([l for l in (_git("status", "--porcelain") or "").splitlines() if l]),
        "code_fingerprint": _code_fingerprint(),
        "resource_fingerprints": _resource_fingerprints(),
        "attacker": attacker,
        "threshold": threshold,
        "success_definition": success_def,
        "eligibility_definition": eligibility_def,
    }


# fields that must match on resume. Includes result_affecting_args (eps_max,
# data_num, ...), code_fingerprint (content hash of result-affecting sources), and
# repro_boundary_versions (torch/cuda/cudnn/transformers/numpy) unless the run
# explicitly opted into best-effort resume (repro_boundary_versions == None).
_RESULT_AFFECTING = ("seed", "checkpoint_sha256", "args_json_sha256",
                     "data_sha256", "attacker", "threshold",
                     "success_definition", "eligibility_definition",
                     "result_affecting_args", "code_fingerprint",
                     "resource_fingerprints", "repro_boundary_versions")


def write_manifest(path, manifest):
    if os.path.exists(path):  # resume: verify compatibility, keep original
        with open(path) as f:
            old = json.load(f)
        diffs = [k for k in _RESULT_AFFECTING if old.get(k) != manifest.get(k)]
        if diffs:
            raise SystemExit(
                f"Manifest mismatch on resume ({', '.join(diffs)}). Use a fresh "
                f"--out_dir for a changed configuration rather than mixing runs.\n"
                f"  existing: {[(k, old.get(k)) for k in diffs]}\n"
                f"  current:  {[(k, manifest.get(k)) for k in diffs]}")
        return old
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path)
    return manifest


class JournalCorruption(Exception):
    """A DURABLE (newline-terminated) record failed to parse/validate. This is
    real corruption, not an interrupted append — recovery stops and preserves the
    evidence rather than silently dropping data."""


def default_validate(rec, seen_indices):
    """Baseline record validation. Raises JournalCorruption on any violation.
    Callers should pass a STRICT validator (make_validator) to Journal so unknown
    statuses can never become silently-terminal."""
    if not isinstance(rec, dict):
        raise JournalCorruption("record is not an object")
    if rec.get("schema") != JOURNAL_SCHEMA:
        raise JournalCorruption(f"wrong/absent schema: {rec.get('schema')!r}")
    idx = rec.get("index")
    if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
        raise JournalCorruption(f"index not a non-negative int: {idx!r}")
    st = rec.get("status")
    if not isinstance(st, str):
        raise JournalCorruption("status not a str")
    if idx in seen_indices:
        raise JournalCorruption(f"duplicate index {idx} (already terminal)")


# --- strict, status-aware validation ---------------------------------------
def _is_nonneg_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def is_finite_nonneg(v):
    """A real, FINITE, non-negative number. Rejects bool, NaN, +/-inf, negatives
    (a modification rate must be a finite fraction >= 0)."""
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(v) and v >= 0)


def _is_sha256(v):
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def make_validator(allowed_statuses, required_by_status, *, index_bound=None,
                   require_source_hash=True):
    """Build a strict record validator.

    * status MUST be in `allowed_statuses` (an UNKNOWN status is corruption, so it
      can never become a silently-terminal record that skews the denominator);
    * `required_by_status[status]` maps field -> predicate; every field must be
      present and satisfy its predicate;
    * `index` bounded to [0, index_bound) when index_bound is given;
    * a valid `source_sha256` is required unless require_source_hash is False.
    """
    def validate(rec, seen_indices):
        default_validate(rec, seen_indices)
        st = rec["status"]
        if st not in allowed_statuses:
            raise JournalCorruption(f"unknown status {st!r} "
                                    f"(allowed: {sorted(allowed_statuses)})")
        if index_bound is not None and not (0 <= rec["index"] < index_bound):
            raise JournalCorruption(f"index {rec['index']} out of range "
                                    f"[0,{index_bound})")
        if require_source_hash and not _is_sha256(rec.get("source_sha256")):
            raise JournalCorruption("missing/invalid source_sha256")
        for field, pred in required_by_status.get(st, {}).items():
            if field not in rec:
                raise JournalCorruption(f"status {st!r} missing '{field}'")
            if not pred(rec[field]):
                raise JournalCorruption(f"status {st!r} field '{field}' invalid: "
                                        f"{rec[field]!r}")
    return validate


# ---------------------------------------------------------------------------
# append-only JSONL journal (source of truth)
# ---------------------------------------------------------------------------
class Journal:
    """Append-only JSONL journal with crash-safe recovery.

    A record is durable once its line ends with a newline (append = write, flush,
    fsync). On load, bytes after the last newline are an interrupted append and are
    truncated; recovery works on bytes so a half-written multi-byte character is
    safe. A durable line that fails to parse or validate, or a duplicate index, is
    corruption: raises JournalCorruption and leaves the file untouched.
    """

    def __init__(self, path, validate=default_validate):
        self.path = path
        self.completed = set()
        self.records = []
        self._validate = validate
        newly_created = not os.path.exists(path)
        if not newly_created:
            self._recover()
        self._fh = open(path, "a", encoding="utf-8")
        if newly_created:
            _fsync_dir(path)  # durably record the file's existence

    def _recover(self):
        with open(self.path, "rb") as f:
            raw = f.read()
        # Durable prefix = everything up to and including the last newline.
        nl = raw.rfind(b"\n")
        durable = raw[:nl + 1] if nl != -1 else b""
        tail_bytes = len(raw) - len(durable)   # interrupted append (any bytes)
        # Only the durable prefix is decoded — never the partial tail.
        text = durable.decode("utf-8", errors="strict")
        for lineno, line in enumerate(text.split("\n"), start=1):
            if line == "":
                continue                        # trailing '' after final '\n'
            try:
                rec = json.loads(line)
                self._validate(rec, self.completed)
            except (json.JSONDecodeError, JournalCorruption) as e:
                # DURABLE line failed -> real corruption. Stop; keep evidence.
                raise JournalCorruption(
                    f"{self.path}: durable record failed at line {lineno}: {e}. "
                    f"Journal preserved for inspection; refusing to auto-repair.")
            self.records.append(rec)
            self.completed.add(rec["index"])
        if tail_bytes:  # truncate the interrupted tail (valid-JSON tail included)
            with open(self.path, "r+b") as f:
                f.truncate(len(durable))
                f.flush(); os.fsync(f.fileno())
            _fsync_dir(self.path)

    def append(self, record):
        self._validate(record, self.completed)
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.records.append(record)
        self.completed.add(record["index"])

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


class EventLog(Journal):
    """Append-only event log where each record is a UNIQUE event (not a terminal
    per-source-index record). Used for the error ledger: a source example may
    error multiple times (once per attempt), so the ledger is keyed by a unique
    auto-incrementing event id (`index`) and additionally records `source_index`.

    Retryability is NOT determined by this log — the caller skips only indices
    present in the RESULTS journal. Resolution = source indices that later get a
    terminal results record. This log never blocks a retry.
    """

    def __init__(self, path, validate):
        super().__init__(path, validate=validate)
        self._next_id = (max(self.completed) + 1) if self.completed else 0

    def record_event(self, event):
        """Append an event; assigns a fresh unique `index` (event id)."""
        event = dict(event)
        event["index"] = self._next_id
        self._next_id += 1
        self.append(event)
        return event["index"]

    def source_indices(self):
        return {r["source_index"] for r in self.records if "source_index" in r}


# Modules present only in the RoChBert graph-fusion model, not in the plain BERT
# or ChineseBERT baselines. Used to reject a baseline checkpoint before
# load_from_checkpoint turns it into a wall of state_dict key errors.
_ROBERT_ONLY_PREFIXES = ("char_embeddings.", "transformer_encoder.", "map_fc.")


def validate_robert_checkpoint(checkpoint_path, model_args):
    """Check a checkpoint is a RoChBert one matching model_args, before building.

    Catches the two failure modes: a baseline checkpoint passed where a RoChBert
    one is needed (missing graph-fusion modules), and a RoChBert checkpoint whose
    saved dimensions disagree with args_json (which can otherwise load silently
    when shapes happen to line up). Raises SystemExit with an actionable message.
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    keys = list(sd.keys())

    missing = [p for p in _ROBERT_ONLY_PREFIXES
               if not any(k.startswith(p) for k in keys)]
    if missing:
        kind = "a ChineseBERT/BERT baseline" if any(
            k.startswith("model.bert.") for k in keys) else "not a RoChBert model"
        raise SystemExit(
            f"{checkpoint_path} looks like {kind}: no "
            f"{', '.join(m.rstrip('.') for m in missing)} weights.\n"
            f"These scripts need a checkpoint from trainer_robert.py. Keep RoChBert "
            f"and baseline checkpoints in separate --save_path directories.")

    # The tokenizer is built from args_json.bert_name, so a checkpoint trained
    # with a different (but same-width) BERT would otherwise be paired with the
    # wrong tokenizer and silently produce wrong token ids.
    _check_bert_provenance(checkpoint_path, ckpt, sd, model_args)

    # Every dimension below is derived from the supplied args, the checkpoint
    # itself and the character resources, exactly as RoBERT.__init__ derives
    # them. args_json does NOT record n_words/vector_size, so anything read from
    # it with a default would silently skip the check.
    expected = _expected_robert_shapes(model_args, sd)

    missing_tensors = [name for name in expected if name not in sd]
    if missing_tensors:
        raise SystemExit(
            f"{checkpoint_path} is missing required weights: "
            f"{', '.join(sorted(missing_tensors))}.\nIt is not a complete "
            f"RoChBert checkpoint from trainer_robert.py.")

    for name, want in sorted(expected.items()):
        got = tuple(sd[name].shape)
        if got != want:
            raise SystemExit(
                f"{name} shape mismatch: checkpoint has {got}, but "
                f"args_node2vec_attn.json and the character resources in "
                f"{getattr(model_args, 'charvec_path', '<charvec_path>')} imply "
                f"{want}.\nThe checkpoint, the args_node2vec_attn.json and the "
                f"res/char_vec files must come from the same training run.")

    # Same-shaped but semantically different character resources would pass every
    # shape check above, so compare the embedding content itself. Training freezes
    # char_embeddings (requires_grad = False), so the saved rows must still equal
    # the vectors the current resources produce.
    _check_char_embedding_content(checkpoint_path, sd, model_args)


def _load_word2id(charvec_path):
    """Rebuild RoBERT's char vocabulary from vocab_advgraph_noweight_109706.txt."""
    word2id = {"unk": 0}
    vocab = os.path.join(charvec_path, "vocab_advgraph_noweight_109706.txt")
    with open(vocab, encoding="utf-8") as f:
        for line in f.readlines():
            word = line.strip()[0]
            if word not in word2id.keys():
                word2id[word] = len(word2id)
    return word2id


_BERT_EMB = "bert_model.embeddings.word_embeddings.weight"


def _saved_hparams(ckpt):
    """The hyperparameters RoBERT.save_hyperparameters(args) stored, if any."""
    hp = ckpt.get("hyper_parameters") if isinstance(ckpt, dict) else None
    if hp is None and isinstance(ckpt, dict):
        hp = ckpt.get("hparams")
    if hp is None:
        return None
    try:
        return dict(hp)
    except (TypeError, ValueError):
        return getattr(hp, "__dict__", None)


def _check_bert_provenance(checkpoint_path, ckpt, sd, model_args):
    """Ensure args_json.bert_name is the BERT the checkpoint was trained with.

    The driver builds its tokenizer from args_json.bert_name. Two same-width
    BERTs (e.g. bert-base-chinese vs chinese-roberta-wwm-ext) pass every shape
    check but have different vocabularies, so the wrong tokenizer would produce
    wrong token ids with no error. RoBERT.save_hyperparameters(args) records the
    name inside the checkpoint, so no network fetch is needed.

    Requires a Lightning checkpoint carrying hyper_parameters. A bare
    torch.save(model.state_dict()) is an unsupported legacy format here: it
    records no bert_name, so the pairing cannot be confirmed.
    """
    arg_name = getattr(model_args, "bert_name", None)
    if arg_name is None:
        raise SystemExit(
            "args_node2vec_attn.json has no bert_name, so the tokenizer cannot "
            "be matched to the checkpoint. Use the file written beside it.")

    hparams = _saved_hparams(ckpt)
    saved_name = (hparams or {}).get("bert_name")

    if saved_name is None:
        # Pre-dates save_hyperparameters, was stripped, or is a bare state_dict
        # saved without Lightning's wrapper. Refuse rather than assume: the
        # exact bert_name comparison below is the only thing that distinguishes
        # two same-width BERTs, and there is nothing here to compare against.
        # The row count is reported for diagnosis only - it is NOT a check, as
        # several models share a vocabulary size.
        vocab_rows = int(sd[_BERT_EMB].shape[0]) if _BERT_EMB in sd else None
        raise SystemExit(
            f"{checkpoint_path} has no saved bert_name (legacy checkpoint), so "
            f"'{arg_name}' cannot be confirmed as the BERT it was trained with.\n"
            f"Its embedding matrix has {vocab_rows} rows. Retrain with the "
            f"current trainer_robert.py, which records bert_name in the "
            f"checkpoint, or verify the pairing yourself and rerun with the "
            f"matching args_node2vec_attn.json.")

    if str(saved_name) != str(arg_name):
        raise SystemExit(
            f"bert_name mismatch: {checkpoint_path} was trained with "
            f"'{saved_name}', but args_node2vec_attn.json says '{arg_name}'.\n"
            f"The tokenizer is built from args_json, so this would pair the "
            f"checkpoint with the wrong vocabulary. Use the args file written "
            f"beside the checkpoint.")

    # bert_name matched, so also confirm the training config in args_json agrees
    # with what the checkpoint recorded.
    for attr, label in (("max_length", "max_length"), ("num_labels", "num_labels")):
        saved_val = (hparams or {}).get(attr)
        arg_val = getattr(model_args, attr, None)
        if saved_val is not None and arg_val is not None and int(saved_val) != int(arg_val):
            raise SystemExit(
                f"{label} mismatch: checkpoint was trained with {saved_val}, "
                f"args_node2vec_attn.json says {arg_val}.\nUse the args file "
                f"written beside the checkpoint.")


def _expected_robert_shapes(model_args, sd):
    """Derive every RoChBert-specific tensor shape from args + resources + ckpt.

    Mirrors RoBERT.__init__: n_words from the vocab file, vector_size from the
    word2vec file, char_size=768 and num_heads/dim_feedforward fixed in the
    model. The BERT hidden size is read from the checkpoint's own embedding
    matrix rather than assumed, so non-768 models are handled correctly.
    """
    import gensim

    charvec_path = getattr(model_args, "charvec_path", None)
    if charvec_path is None:
        raise SystemExit(
            "args_node2vec_attn.json has no charvec_path, so the checkpoint "
            "cannot be validated. Use the file written beside the checkpoint.")
    max_len = int(model_args.max_length)
    num_labels = int(model_args.num_labels)

    word2id = _load_word2id(charvec_path)
    n_words = max(word2id.values()) + 1
    kv = gensim.models.KeyedVectors.load_word2vec_format(
        os.path.join(charvec_path, "word2vec_advgraph_noweight_109706.txt"),
        binary=False, unicode_errors="ignore")
    vec = int(kv.vector_size)

    char_size = 768        # RoBERT.char_size, fixed in the model
    if _BERT_EMB not in sd:
        raise SystemExit(
            f"Checkpoint has no {_BERT_EMB}, so the BERT hidden size cannot be "
            f"determined. It is not a complete RoChBert checkpoint.")
    hidden = int(sd[_BERT_EMB].shape[1])   # derived, not assumed
    fused = hidden + char_size

    return {
        "char_embeddings.weight": (n_words, vec),
        # nn.TransformerEncoderLayer(d_model=vec, nhead=2)
        "transformer_encoder.layers.0.self_attn.in_proj_weight": (3 * vec, vec),
        "transformer_encoder.layers.0.linear1.weight": (2048, vec),
        "transformer_encoder.layers.0.linear2.weight": (vec, 2048),
        # map_fc collapses vec * max_len -> char_size, so it pins max_length.
        "map_fc.weight": (char_size, vec * max_len),
        # Second encoder runs on the fused BERT+char representation.
        "transformer_encoder2.layers.0.self_attn.in_proj_weight": (3 * fused, fused),
        "transformer_encoder2.layers.0.linear1.weight": (1024, fused),
        "classifier.weight": (num_labels, fused * max_len),
    }


def _check_char_embedding_content(checkpoint_path, sd, model_args):
    """Verify the frozen char embeddings match the supplied resources.

    Catches resources that share the checkpoint's shapes but hold different
    vectors, which no shape comparison can detect.
    """
    import numpy as np
    import gensim

    charvec_path = model_args.charvec_path
    word2id = _load_word2id(charvec_path)
    kv = gensim.models.KeyedVectors.load_word2vec_format(
        os.path.join(charvec_path, "word2vec_advgraph_noweight_109706.txt"),
        binary=False, unicode_errors="ignore")

    saved = sd["char_embeddings.weight"]
    saved = saved.detach().cpu().numpy() if hasattr(saved, "detach") else np.asarray(saved)

    # Rows absent from the word2vec file were left at their random init, so only
    # compare rows the resources actually determine.
    checked = mismatched = 0
    for word, idx in word2id.items():
        try:
            vector = kv[word]
        except KeyError:
            continue
        checked += 1
        if not np.allclose(saved[idx], vector, atol=1e-5):
            mismatched += 1
            if mismatched > 8:
                break

    if checked == 0:
        raise SystemExit(
            f"No character in {charvec_path} matched the word2vec file, so the "
            f"checkpoint's embeddings cannot be validated. The res/char_vec "
            f"files look inconsistent with each other.")
    if mismatched:
        raise SystemExit(
            f"{checkpoint_path} was trained with different character vectors "
            f"than the ones in {charvec_path} (at least {mismatched} of "
            f"{checked} rows differ, same shape).\nUse the res/char_vec "
            f"resources from the original training run.")


def atomic_write_lines(path, lines):
    """Materialize a text file atomically (write .tmp + fsync + rename + fsync dir)."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path)


def atomic_write_json(path, obj):
    """Atomically write a JSON file (used for summary.json etc.)."""
    atomic_write_lines(path, [json.dumps(obj, indent=2, ensure_ascii=False)])
