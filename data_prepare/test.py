if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset(
        "/data/personal/nus-zxl/VerticalMoE/data_prepare/redpajama_dataset.py",
        "c4",
        trust_remote_code=True,
        split="train",
    )
    ds = ds.train_test_split(test_size=0.2, seed=42)
    print(ds)
