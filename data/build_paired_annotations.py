import json
from pathlib import Path

import pyarrow.parquet as pq


dataset_dir = Path(__file__).resolve().parent / "VideoDetailCaption"
parquet_path = dataset_dir / "data" / "test-00000-of-00001.parquet"
high_dir = dataset_dir / "paired" / "high"
low_dir = dataset_dir / "paired" / "low"
output_path = dataset_dir / "paired" / "annotations.jsonl"

# 讀取原始 GT
table = pq.read_table(parquet_path)
gt_by_name = {
    row["video_name"]: row
    for row in table.to_pylist()
}

# 取得 High 與 Low 的影片
high_files = sorted(
    path for path in high_dir.iterdir()
    if path.suffix.lower() in {".mp4", ".mkv"}
)
low_files = sorted(low_dir.glob("*.mp4"))

high_by_name = {path.stem: path for path in high_files}
low_by_name = {path.stem: path for path in low_files}

# 驗證配對
high_names = set(high_by_name)
low_names = set(low_by_name)

if len(high_names) != 80:
    raise RuntimeError(f"Expected 80 High videos, found {len(high_names)}")

if len(low_names) != 80:
    raise RuntimeError(f"Expected 80 Low videos, found {len(low_names)}")

if high_names != low_names:
    raise RuntimeError(
        f"High/Low mismatch: "
        f"missing Low={sorted(high_names - low_names)}, "
        f"missing High={sorted(low_names - high_names)}"
    )

missing_gt = high_names - set(gt_by_name)
if missing_gt:
    raise RuntimeError(f"Missing GT: {sorted(missing_gt)}")

# 建立 80 筆配對標註
records = []

for video_name in sorted(high_names):
    gt = gt_by_name[video_name]

    records.append(
        {
            "video_name": video_name,
            "high_path": str(
                high_by_name[video_name].relative_to(dataset_dir)
            ),
            "low_path": str(
                low_by_name[video_name].relative_to(dataset_dir)
            ),
            "question": gt["question"],
            "answer": gt["answer"],
        }
    )

# 輸出 JSONL
with output_path.open("w", encoding="utf-8") as file:
    for record in records:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Created: {output_path}")
print(f"Records: {len(records)}")