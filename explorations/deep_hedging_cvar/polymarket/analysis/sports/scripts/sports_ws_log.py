"""Log Polymarket's public sports score websocket (wss://sports-api.polymarket.com/ws) with local receive times.
No subscription frame is needed; the server pings ('ping' text) every 5 s and expects 'pong'.
Usage: sports_ws_log.py OUT.jsonl DURATION_S"""
import asyncio, json, sys, time
import websockets

URL = "wss://sports-api.polymarket.com/ws"


async def main(out, dur):
    t_end = time.time() + dur
    with open(out, "a") as fh:
        while time.time() < t_end:
            try:
                async with websockets.connect(URL, ping_interval=None, open_timeout=15, max_size=2**22) as ws:
                    fh.write(json.dumps({"r": time.time(), "_connect": True}) + "\n")
                    while time.time() < t_end:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                        r = time.time()
                        if msg in ("ping", "PING"):
                            await ws.send("pong"); continue
                        fh.write(json.dumps({"r": round(r, 3), "m": msg if isinstance(msg, str) else msg.decode()}) + "\n"); fh.flush()
            except Exception as e:  # noqa: BLE001
                fh.write(json.dumps({"r": time.time(), "_error": repr(e)[:200]}) + "\n"); fh.flush()
                await asyncio.sleep(3)

asyncio.run(main(sys.argv[1], float(sys.argv[2])))
