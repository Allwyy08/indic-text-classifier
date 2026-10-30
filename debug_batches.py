import sys
import yaml
import traceback
from src.train_utils import prepare_hf_dataset
from src.model_utils import load_tokenizer, create_classification_model
from src.data_utils import load_splits

print("=" * 70)
print("DEBUGGING BATCH 20-30 OF SENTIMENT DATASET")
print("=" * 70)

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)

model_name = config["model"]["pretrained_model_name"]
tokenizer = load_tokenizer(model_name)
splits = load_splits("data/sentiment")
train_df = splits["train"]
label2id = {str(k): int(v) for k, v in config["tasks"]["sentiment"]["label2id"].items()}

print(f"Total train_df rows: {len(train_df)}")
train_ds = prepare_hf_dataset(train_df, tokenizer, label2id)
print("Tokenized train dataset successfully, length:", len(train_ds))

# Check samples around batch 24 (steps 20-30 * 16 = indices 320 to 480)
print("\nInspecting samples around index 300 to 450:")
for idx in range(350, 400):
    try:
        sample = train_ds[idx]
        text = train_df.iloc[idx]["text"]
        label = train_df.iloc[idx]["label"]
        tokens = sample["input_ids"]
        if len(tokens) == 0:
            print(f"EMPTY TOKENS at {idx}!")
    except Exception as e:
        print(f"ERROR at idx {idx}: {e}")
        traceback.print_exc()

print("Inspection completed without Python errors!")

