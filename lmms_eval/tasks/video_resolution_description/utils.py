from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPO_ROOT / "data" / "VideoDetailCaption"


def _video_path(doc, path_key):
    path = DATASET_ROOT / doc[path_key]

    if not path.exists():
        raise FileNotFoundError(path)

    return str(path.resolve())


def high_doc_to_visual(doc):
    return [_video_path(doc, "high_path")]


def low_doc_to_visual(doc):
    return [_video_path(doc, "low_path")]


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return doc["question"]


def doc_to_target(doc):
    return doc["answer"]


def _doc_to_messages(doc, path_key):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "url": _video_path(doc, path_key),
                },
                {
                    "type": "text",
                    "text": doc["question"],
                },
            ],
        }
    ]


def high_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return _doc_to_messages(doc, "high_path")


def low_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return _doc_to_messages(doc, "low_path")


def process_results(doc, results):
    prediction = results[0] if results else ""

    # 暫時只用來確認模型有成功產生描述，不是最終品質分數。
    return {
        "word_count": len(prediction.split()),
    }