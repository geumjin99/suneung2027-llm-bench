"""Download 2027 CSAT question papers and answer keys from KICE's exam-day CDN."""
import os, urllib.request
BASE = "https://cdn.kice.re.kr/27dae-suneung/"
FILES = [  # 1=Korean, 2=Math, 3=English, 31=English listening script; 'a' suffix = answer key
    "27DaeSuneung_1.pdf", "27DaeSuneung_1a.pdf", "27DaeSuneung_2.pdf", "27DaeSuneung_2a.pdf",
    "27DaeSuneung_3.pdf", "27DaeSuneung_3a.pdf", "27DaeSuneung_31.pdf",
]
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "pdf")
os.makedirs(out, exist_ok=True)
for f in FILES:
    dst = os.path.join(out, f)
    if os.path.exists(dst):
        continue
    try:
        req = urllib.request.Request(BASE + f, headers={"User-Agent": "Mozilla/5.0"})
        open(dst, "wb").write(urllib.request.urlopen(req, timeout=60).read())
        print("ok", f)
    except Exception as e:
        print("skip", f, e)
