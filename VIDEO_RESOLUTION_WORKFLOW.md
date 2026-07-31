# 影片解析度描述實驗操作

## 1. 進入開發環境

每次開啟新的 terminal 後執行：

```bash
cd /home/Henry/lmms-eval
source .venv/bin/activate
```

執行後，terminal 前方應出現 `(.venv)`。

## 2. 資料與標註位置

原始 VideoDetailCaption 資料：

```text
data/VideoDetailCaption/extracted/Test_Videos/       原始影片
data/VideoDetailCaption/data/test-00000-of-00001.parquet
```

目前使用 80 組配對資料：

```text
data/VideoDetailCaption/paired/high/                 80 支 1920×1080 原片的 symbolic links
data/VideoDetailCaption/paired/low/                  80 支 432×240 低畫質 MP4
data/VideoDetailCaption/paired/annotations.jsonl     80 筆 question、GT 與 High/Low 路徑
```

High 與 Low 使用相同影片內容及相同人工 GT。Low 的視訊設定為
`432×240`、H.264、約 `400 kbps`，音訊為 AAC、約 `96 kbps`。

## 3. 建立 Low 與配對標註

### High 選擇方式

- 使用 `ffprobe` 檢查原始 MP4/MKV 的第一條視訊串流。
- 只選 coded resolution **剛好為 `1920×1080`** 的影片，共 80 支。
- `paired/high` 使用 symbolic links 指向原片，不重新編碼，因此保留原始畫質、
  FPS、bitrate、容器及音訊。

### Low 轉換方式

將上述 Full HD 原片轉成 Low；已存在的輸出會自動略過：

```bash
python data/convert_fhd_to_low.py
```

轉換時保留原始長寬比，必要時補黑邊至 `432×240`，不主動變更 FPS；輸出統一為
MP4、H.264、約 `400 kbps`、YUV 4:2:0，音訊為 AAC、約 `96 kbps`。原片不會被
覆寫，High 與 Low 使用相同檔名 stem 進行一對一配對。

### 配對與 GT

根據 `paired/high`、`paired/low` 和原始 Parquet 建立 80 筆配對標註：

```bash
python data/build_paired_annotations.py
```

腳本會確認 High 與 Low 各有 80 支、檔名完全對應，而且每支影片都能在原始
Parquet 找到 question 與 GT；兩種解析度共用同一組 question 和 GT。

## 4. 模型與影片輸入方式

目前使用三個模型：

```text
Baseline：Qwen/Qwen3-VL-4B-Instruct
較新模型：Qwen/Qwen3.5-4B
較大模型：Qwen/Qwen3.5-9B
```

三者都透過自訂的 `qwen3_vl_native_video` wrapper 執行：

```text
lmms_eval/models/simple/qwen3_vl_native_video.py
```

共同設定：

- 單張 GPU 1
- 每支影片均勻抽取 12 frames
- High 保留 1920×1080，Low 保留 432×240
- `batch_size=1`
- `max_new_tokens=768`
- `temperature=0`
- High 與 Low 使用相同 prompt、GT 及時間抽樣方式

自訂 lmms-eval tasks 位於：

```text
lmms_eval/tasks/video_resolution_description/
```

## 5. 產生 80 組影片描述

### Qwen3-VL-4B baseline

```bash
CUDA_VISIBLE_DEVICES=1 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
python -m lmms_eval \
  --model qwen3_vl_native_video \
  --model_args pretrained=Qwen/Qwen3-VL-4B-Instruct,device_map=auto,attn_implementation=sdpa,native_num_frames=12 \
  --tasks video_resolution_high,video_resolution_low \
  --batch_size 1 \
  --predict_only \
  --log_samples \
  --output_path outputs/qwen3_vl_4b_native_resolution_full80
```

### Qwen3.5-4B

```bash
CUDA_VISIBLE_DEVICES=1 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
python -m lmms_eval \
  --model qwen3_vl_native_video \
  --model_args pretrained=Qwen/Qwen3.5-4B,device_map=auto,attn_implementation=sdpa,native_num_frames=12,enable_thinking=false \
  --tasks video_resolution_high,video_resolution_low \
  --batch_size 1 \
  --predict_only \
  --log_samples \
  --output_path outputs/qwen35_4b_native_resolution_full80
```

### Qwen3.5-9B

```bash
CUDA_VISIBLE_DEVICES=1 \
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m lmms_eval \
  --model qwen3_vl_native_video \
  --model_args pretrained=Qwen/Qwen3.5-9B,device_map=auto,attn_implementation=sdpa,native_num_frames=12,enable_thinking=false \
  --tasks video_resolution_high,video_resolution_low \
  --batch_size 1 \
  --predict_only \
  --log_samples \
  --output_path outputs/qwen35_9b_native_resolution_full80
```

Qwen3.5-9B 在 RTX 4090 上執行時約使用 `22081 MiB / 24564 MiB`，因此需維持
`batch_size=1`、12 frames，並避免同一張 GPU 同時執行其他程式。

每次完整實驗會產生 80 筆 High 與 80 筆 Low 描述，共 160 筆。`--predict_only`
只產生並保存描述，不在這一步呼叫付費評分 API。

## 6. 使用 GT 評分描述

專案根目錄的 `.env` 保存評分 API 設定：

```dotenv
OPENAI_API_KEY=填入自己的_API_key
MODEL_VERSION=gpt-4o-2024-11-20
```

不要將 API key 貼到對話、程式碼或 Git。評分腳本會自動讀取 `.env`，使用 repo
原有的 Video Detail Description judge，根據 question、人工 GT 和模型描述給予 `0–5`
分。

評分 Qwen3-VL-4B：

```bash
python data/evaluate_video_resolution_outputs.py \
  outputs/qwen3_vl_4b_native_resolution_full80
```

評分 Qwen3.5-4B：

```bash
python data/evaluate_video_resolution_outputs.py \
  outputs/qwen35_4b_native_resolution_full80
```

評分 Qwen3.5-9B：

```bash
python data/evaluate_video_resolution_outputs.py \
  outputs/qwen35_9b_native_resolution_full80
```

評分會逐筆保存並支援中斷續跑。結果位於各實驗目錄下：

```text
gpt_eval_scores.json
```

查看 High 與 Low 平均分數：

```bash
jq '.averages' outputs/<實驗目錄>/gpt_eval_scores.json
```

## 7. 主要輸出目錄

```text
outputs/qwen3_vl_4b_native_resolution_full80/
outputs/qwen35_4b_native_resolution_full80/
outputs/qwen35_9b_native_resolution_full80/
```

每個目錄中的 `samples_video_resolution_high.jsonl` 與
`samples_video_resolution_low.jsonl` 保存逐支影片的模型描述及 GT；
`gpt_eval_scores.json` 保存逐筆分數與 High/Low 平均分數。

## 8. 目前完整 80 支結果

評分模型皆為 `gpt-4o-2024-11-20`，分數範圍為 `0–5`：

| 生成模型 | High | Low | High − Low |
|---|---:|---:|---:|
| Qwen3-VL-4B-Instruct | 3.2625 | 2.9000 | 0.3625 |
| Qwen3.5-4B | 3.4500 | 3.1625 | 0.2875 |
| Qwen3.5-9B | 3.0125 | 3.0125 | 0.0000 |

Qwen3.5-9B 的逐片配對結果為 High 勝 15 支、Low 勝 15 支、平手 50 支；High
與 Low 各得 241 分。這表示在目前的 12-frame native-resolution 流程與獨立
`0–5` 詳細度評分下，9B 沒有呈現整體 High 優於 Low，但個別影片仍有明顯差異。
此評分主要衡量描述的完整度與具體程度，且 High/Low 是分開評分，因此不能單獨
代表事實正確性或幻覺程度。

## 9. 下一個推薦資料集：VDC

VDC（Video Detailed Caption）專門評估詳細影片描述，比目前的
VideoDetailCaption 更貼近本實驗主題。資料集約有 1,027 支影片、總大小約 80 GB，
每支影片提供多種結構化描述：

- `short`：一句整體摘要
- `background`：背景、地點、天氣與物件
- `main_object`：主要人物或物件的屬性與動作
- `camera`：鏡頭角度、移動與轉場
- `detailed`：完整詳細描述

VDCScore 會把 GT 拆成多個短 QA，再檢查模型描述能否回答這些問題，比只給一個
`0–5` 詳細度分數更細緻。相關資源：[VDC 論文](https://arxiv.org/abs/2410.03051)、
[VDC 資料集](https://huggingface.co/datasets/wchai/Video-Detailed-Caption)。

目前 lmms-eval 已內建 `lmms_eval/tasks/vdc`，包含：

```text
detailed_test
background_test
main_object_test
camera_test
short_test
```

其評分器可使用本機 `Llama-3.1-8B-Instruct` 搭配 SGLang。生成模型與評分模型可
分開執行，因此單張 RTX 4090 可以先完成影片描述，再卸載生成模型並執行評分。

使用前仍需先檢查影片實際解析度，不能假設全部都是 1080p。預計沿用目前流程：

1. 篩選 Full HD 影片。
2. 保留原片作為 High。
3. 轉成 `432×240` 作為 Low。
4. High 與 Low 共用相同 GT、prompt 與時間抽樣方式。
