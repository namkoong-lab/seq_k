import json, glob, textwrap, sys

key = sys.argv[1] if len(sys.argv) > 1 else "0e073591"
run = sys.argv[2] if len(sys.argv) > 2 else "healthbench.seqk.binary"
ats = []
for f in glob.glob(f"runs/{run}/tasks/*.json"):
    d = json.load(open(f))
    if key in d["task_id"]:
        ats.append(d)
ats.sort(key=lambda a: a["attempt_index"])


def w(s, n=3, width=96):
    return "\n      ".join(textwrap.wrap(" ".join(str(s).split()), width)[:n])


print("QUESTION:", ats[0]["task_prompt"].split("# Conversation")[1].split("Write the")[0].strip())
for a in ats:
    g = a["result"]["judge_details"]["rubric_grades"]
    metpos = [x for x in g if x["points"] > 0 and x["criteria_met"]]
    misspos = [x for x in g if x["points"] > 0 and not x["criteria_met"]]
    negtrig = [x for x in g if x["points"] < 0 and x["criteria_met"]]
    print(f"\n===== ATTEMPT {a['attempt_index']+1}  score={a['result']['score']:.2f} =====")
    print("RESPONSE:", w(a["output"], 4))
    print("  met positive pts:", sum(x["points"] for x in metpos),
          "of", sum(x["points"] for x in g if x["points"] > 0))
    print("  MISSED positive:", [f"+{int(x['points'])}:{x['criterion'][:48]}" for x in misspos])
    print("  triggered negatives:", [f"{int(x['points'])}:{x['criterion'][:42]}" for x in negtrig])
