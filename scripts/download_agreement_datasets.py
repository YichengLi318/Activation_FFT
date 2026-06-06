import os
from typing import Optional

from datasets import DatasetDict, load_dataset


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(ROOT, "dataset")


def _save_hf_dataset(
    dataset_name: str,
    out_dir: str,
    config_name: Optional[str] = None,
) -> bool:
    try:
        print(f"[download] {dataset_name} config={config_name}", flush=True)
        if config_name:
            ds = load_dataset(dataset_name, config_name)
        else:
            ds = load_dataset(dataset_name)
        os.makedirs(out_dir, exist_ok=True)
        ds.save_to_disk(out_dir)
        print(f"[ok] saved to {out_dir}", flush=True)
        return True
    except Exception as exc:
        print(f"[fail] {dataset_name} config={config_name}: {exc}", flush=True)
        return False


def download_blimp() -> bool:
    out_dir = os.path.join(DATASET_DIR, "blimp")
    try:
        print("[download] BLiMP regular_plural_subject_verb_agreement_1", flush=True)
        ds = load_dataset("nyu-mll/blimp", "regular_plural_subject_verb_agreement_1")
        os.makedirs(out_dir, exist_ok=True)
        ds.save_to_disk(out_dir)
        print(f"[ok] saved to {out_dir}", flush=True)
        return True
    except Exception as exc:
        print(f"[fail] BLiMP: {exc}", flush=True)
        return False


def download_mawps() -> bool:
    out_dir = os.path.join(DATASET_DIR, "mawps")
    try:
        print("[download] MAWPS", flush=True)
        ds = load_dataset("garrethlee/MAWPS")
        train = ds["train"]
        test = ds["test"] if "test" in ds else None
        split = train.train_test_split(test_size=0.1, seed=42)
        dd = DatasetDict({"train": split["train"], "validation": split["test"]})
        if test is not None:
            dd["test"] = test
        os.makedirs(out_dir, exist_ok=True)
        dd.save_to_disk(out_dir)
        print(f"[ok] saved to {out_dir}", flush=True)
        return True
    except Exception as exc:
        print(f"[fail] MAWPS: {exc}", flush=True)
        return False


def main():
    os.makedirs(DATASET_DIR, exist_ok=True)

    ok_blimp = download_blimp()
    ok_copa = _save_hf_dataset("super_glue", os.path.join(DATASET_DIR, "copa"), "copa")
    ok_boolq = _save_hf_dataset("super_glue", os.path.join(DATASET_DIR, "boolq"), "boolq")
    ok_mawps = download_mawps()
    ok_ppl = _save_hf_dataset("wikitext", os.path.join(DATASET_DIR, "ppl_wikitext2"), "wikitext-2-raw-v1")

    print("\n=== summary ===", flush=True)
    print(f"BLiMP: {'ok' if ok_blimp else 'fail'}", flush=True)
    print(f"COPA: {'ok' if ok_copa else 'fail'}", flush=True)
    print(f"BoolQ: {'ok' if ok_boolq else 'fail'}", flush=True)
    print(f"MAWPS: {'ok' if ok_mawps else 'fail'}", flush=True)
    print(f"PPL corpus (WikiText2): {'ok' if ok_ppl else 'fail'}", flush=True)

    if not (ok_blimp and ok_copa and ok_boolq and ok_mawps and ok_ppl):
        raise RuntimeError("One or more datasets failed to download. Check logs above.")


if __name__ == "__main__":
    main()
