import os

root = r"C:\Users\FPT SHOP\OneDrive\Documents\XLA\DETECTPSandExtract"

files_to_create = {
    "src": ["dataset.py", "transforms.py", "model.py", "train.py", "evaluate.py", "predict.py"],
    ".": ["run_training.py", "run_predict.py", "config.yaml", "requirements.txt", "README.md"]
}

for folder, files in files_to_create.items():
    base = os.path.join(root, folder)
    os.makedirs(base, exist_ok=True)
    for fname in files:
        fpath = os.path.join(base, fname)
        if not os.path.exists(fpath):
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(f"# {fname} - khởi tạo\n")
            print(f"Đã tạo: {fpath}")
