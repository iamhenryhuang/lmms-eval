# 影片描述解析度實驗

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

不要將 API key 貼到對話、程式碼或 Git。生成與評分是完全分離的兩個階段，因此可
重跑評分而不必重新執行 GPU 生成。目前保留兩種互補的評分方法，兩者都只看
question、人工 GT 與模型描述，不會直接觀看影片。

### 6.1 獨立 VDD 評分

`evaluate_video_resolution_outputs.py` 使用 repo 原有的 Video Detail Description
judge，分別對每一筆 High 和 Low 描述給予 `0–5` detail orientation 分數，主要衡量
GT 重點涵蓋程度與描述具體程度。High/Low 是兩次獨立 API 呼叫，judge 不知道另一個
版本的回答。

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

### 6.2 雙順序 Pairwise 評分

`evaluate_video_resolution_pairwise.py` 直接比較同一支影片的 High 與 Low 描述。
每次 API 呼叫會提供 question、GT、Candidate A 和 Candidate B，但不告訴 judge 哪個
是 High。完整評估包含兩輪：

1. 第一輪以固定 seed 平衡 A/B 位置，High 與 Low 各有 40 次位於 Candidate A。
2. 第二輪交換每一筆的 A/B 順序，以降低位置偏誤。

Judge 只輸出 `A`、`B` 或 `tie` 以及簡短理由，並根據正確性、GT 重點涵蓋程度、有效
細節、矛盾與重複內容判斷；不能只因回答較長就判定勝出。最終採嚴格共識：兩輪都
選 High 才算 High 勝，兩輪都選 Low 才算 Low 勝，其餘判定為 Tie/conflict。

Pairwise 評分 Qwen3-VL-4B：

```bash
python data/evaluate_video_resolution_pairwise.py \
  outputs/qwen3_vl_4b_native_resolution_full80
```

Pairwise 評分 Qwen3.5-4B：

```bash
python data/evaluate_video_resolution_pairwise.py \
  outputs/qwen35_4b_native_resolution_full80
```

Pairwise 評分 Qwen3.5-9B：

```bash
python data/evaluate_video_resolution_pairwise.py \
  outputs/qwen35_9b_native_resolution_full80
```

每完成一筆就會原子寫入進度，中斷後使用同一指令即可跳過已完成項目。結果位於：

```text
gpt_pairwise_scores.json
```

主要欄位：

```text
results                 第一輪結果
swapped_results         A/B 交換後的第二輪結果
first_pass_summary      第一輪摘要
swapped_pass_summary    第二輪摘要
consensus_results       逐片嚴格共識結果
consensus_summary       最終 High/Low/Tie 統計
```

查看最終共識摘要：

```bash
jq '.consensus_summary' outputs/<實驗目錄>/gpt_pairwise_scores.json
```

### 6.3 使用 GPT-5.1 交叉驗證 Pairwise 結果

為了檢查結論是否過度依賴單一 judge，另外使用固定版本
`gpt-5.1-2025-11-13` 重跑相同的雙順序 Pairwise 評分。評分 prompt、A/B 分配、
交換順序與嚴格共識規則皆保持不變；GPT-5.1 使用
`max_completion_tokens=200`、`reasoning_effort=none`。原本 GPT-4o 結果不會被
覆蓋。

三組完整結果可依序執行：

```bash
for experiment in \
  outputs/qwen35_4b_native_resolution_full80 \
  outputs/qwen35_9b_native_resolution_full80 \
  outputs/qwen3_vl_4b_native_resolution_full80
do
  MODEL_VERSION=gpt-5.1-2025-11-13 \
  python data/evaluate_video_resolution_pairwise.py \
    "$experiment" \
    --output "$experiment/gpt_pairwise_scores_gpt51.json"
done
```

每組包含 80 組影片並交換 A/B，因此會進行 160 次 API 評分。結果逐筆保存且支援
中斷續跑，最終摘要位於：

```text
outputs/<實驗目錄>/gpt_pairwise_scores_gpt51.json
```

## 7. 主要輸出目錄

```text
outputs/qwen3_vl_4b_native_resolution_full80/
outputs/qwen35_4b_native_resolution_full80/
outputs/qwen35_9b_native_resolution_full80/
```

每個目錄中的 `samples_video_resolution_high.jsonl` 與
`samples_video_resolution_low.jsonl` 保存逐支影片的模型描述及 GT；
`gpt_eval_scores.json` 保存獨立 VDD 分數與 High/Low 平均分數；
`gpt_pairwise_scores.json` 保存 GPT-4o 的兩輪盲測結果；
`gpt_pairwise_scores_gpt51.json` 保存 GPT-5.1 的相同評分與嚴格共識摘要。

## 8. 目前完整 80 支結果

評分模型皆為 `gpt-4o-2024-11-20`。Pairwise High win rate 排除 Tie/conflict：

| 生成模型 | VDD High | VDD Low | High − Low | Pairwise High | Low | Tie/conflict | High win rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-4B-Instruct | 3.2625 | 2.9000 | 0.3625 | 24 | 12 | 44 | 66.7% |
| Qwen3.5-4B | 3.4500 | 3.1625 | 0.2875 | 20 | 9 | 51 | 69.0% |
| Qwen3.5-9B | 3.0125 | 3.0125 | 0.0000 | 19 | 17 | 44 | 52.8% |

兩個 4B 模型在 VDD 與 Pairwise 中都呈現 High 較好的趨勢，但嚴格共識後仍有大量
Tie/conflict，因此不能宣稱差異很大。Qwen3.5-9B 在兩種評分方法下都幾乎沒有整體
High/Low 差異。這表示解析度影響可能與模型及描述策略有關，並非所有模型都一致；
9B 的無差異也不等於它整體描述品質一定比 4B 更好。

兩種評分都屬於 reference-based text evaluation：judge 看不到影片，也無法確認 GT
沒有提到的額外細節是真是假。因此目前結果應解讀為「解析度影響模型描述與 GT 的
吻合程度」，而不是完整的影片事實驗證。

### GPT-5.1 Pairwise 交叉驗證

固定 judge 為 `gpt-5.1-2025-11-13`，其餘 Pairwise 設定與 GPT-4o 完全相同：

| 生成模型 | Pairwise High | Low | Tie/conflict | High win rate |
|---|---:|---:|---:|---:|
| Qwen3-VL-4B-Instruct | 35 | 18 | 27 | 66.0% |
| Qwen3.5-4B | 33 | 14 | 33 | 70.2% |
| Qwen3.5-9B | 33 | 19 | 28 | 63.5% |

GPT-5.1 比 GPT-4o 更常選出勝者，因此三組的 Tie/conflict 都減少；兩個 judge 直接
將 High/Low 勝負完全反轉的案例很少，主要差異是 GPT-4o 判定為 Tie 的部分案例被
GPT-5.1 判為 High 或 Low。三個模型在 GPT-5.1 下皆呈現 High 勝多於 Low 勝，顯示
高解析度具有一致但非全面性的優勢；仍有約 35%–41% 的影片屬於 Tie/conflict，不能
解讀為每支影片都會因解析度提高而改善。

GPT-5.1 很少在單輪直接回答 Tie，但交換 A/B 後仍有部分判斷不一致，因此保留雙順序
評分與嚴格共識是必要的。綜合兩個 judge，較適當的結論是：固定抽取 12 frames 時，
高解析度通常較容易產生符合 GT 的描述，但整體場景與大型動作在 240p 中仍多半可辨識，
所以差異屬中等程度，而非壓倒性差距。

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
