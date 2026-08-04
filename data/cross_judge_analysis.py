import json, sys
from pathlib import Path

exp = Path(sys.argv[1])
j4o = json.loads((exp / "gpt_pairwise_scores.json").read_text())
j51 = json.loads((exp / "gpt_pairwise_scores_gpt51.json").read_text())

c4o = {r["doc_id"]: r["winner_quality"] for r in j4o["consensus_results"]}
c51 = {r["doc_id"]: r["winner_quality"] for r in j51["consensus_results"]}

both_high = [d for d in c4o if c4o[d] == "high" and c51.get(d) == "high"]
both_low  = [d for d in c4o if c4o[d] == "low"  and c51.get(d) == "low"]
flip      = [d for d in c4o if {"high","low"} == {c4o[d], c51.get(d)}]
tie_to_hi = [d for d in c4o if c4o[d] == "tie" and c51.get(d) == "high"]

print(f"兩個 judge 都判 High 勝: {len(both_high)}  -> {sorted(both_high)}")
print(f"兩個 judge 都判 Low 勝:  {len(both_low)}  -> {sorted(both_low)}")
print(f"判決完全相反:            {len(flip)}  -> {sorted(flip)}")
print(f"4o平手但5.1判High:       {len(tie_to_hi)}  -> {sorted(tie_to_hi)}")
