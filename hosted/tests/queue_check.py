"""Send several rewrites at once and confirm they all finish through the queue.

Start the app first (see browser_check.py), then: python tests/queue_check.py [url]
Works against the public Space too: python tests/queue_check.py https://<user>-<space>.hf.space
"""
import sys
import threading
import time

from gradio_client import Client

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
PASSAGE = " ".join(["I spent the weekend fixing a bug in our class project and learned a lot about debugging."] * 7)
results, started = {}, time.perf_counter()


def one(i):
    client = Client(URL, verbose=False)
    t0 = time.perf_counter()
    job = client.submit(PASSAGE + f" Request {i}.", "basic", False, api_name="/rewrite_adapter")
    seen_queue = False
    while not job.done():
        status = job.status()
        if status.code.name == "IN_QUEUE":
            seen_queue = True
        time.sleep(0.5)
    out = job.outputs()[-1]
    results[i] = dict(seconds=round(time.perf_counter() - t0, 1), queued=seen_queue,
                      status=out[1], rewritten=bool(out[0]))


threads = [threading.Thread(target=one, args=(i,)) for i in range(N)]
for t in threads:
    t.start()
for t in threads:
    t.join()
for i in sorted(results):
    print(i, results[i])
ok = all(r["rewritten"] and r["status"].startswith("Revision ready") for r in results.values())
print("all finished:", ok, "| wall time", round(time.perf_counter() - started, 1), "s")
