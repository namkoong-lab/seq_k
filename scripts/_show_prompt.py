import json, glob, re, sys

key = sys.argv[1] if len(sys.argv) > 1 else "0e073591"
run = sys.argv[2] if len(sys.argv) > 2 else "healthbench.seqk.binary"
att = sys.argv[3] if len(sys.argv) > 3 else "04"
path = [f for f in glob.glob(f"runs/{run}/tasks/*_attempt_{att}.json") if key in f][0]
p = json.load(open(path))["prompt"]


# collapse <PreviousAttempt> bodies so structure + feedback are readable
def collapse(m):
    body = " ".join(m.group(1).split())
    return f"<PreviousAttempt>\n  {body[:120]} ...[truncated]\n</PreviousAttempt>"


p = re.sub(r"<PreviousAttempt>(.*?)</PreviousAttempt>", collapse, p, flags=re.S)
print(p)
