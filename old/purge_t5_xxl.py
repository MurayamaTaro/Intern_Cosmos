# purge_t5_xxl.py
#!/usr/bin/env python3
import argparse, shutil
from pathlib import Path

ROOT = Path("dataset_vript")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="確認なしで削除")
    args = ap.parse_args()

    targets = sorted((p for p in ROOT.glob("*/t5_xxl") if p.is_dir()))
    if not targets:
        print("[INFO] no t5_xxl dirs found.")
        return
    print("[INFO] found t5_xxl dirs:")
    for p in targets:
        print("  -", p)

    if not args.yes:
        print("\n[DRY-RUN] add --yes to actually delete.")
        return

    # もう一度聞く（事故防止）
    ans = input("Delete ALL above directories? type 'yes' to continue: ").strip().lower()
    if ans != "yes":
        print("[CANCELLED]")
        return

    for p in targets:
        shutil.rmtree(p, ignore_errors=True)
        print("[DEL]", p)
    print("[DONE] all t5_xxl removed.")

if __name__ == "__main__":
    main()
