import threading

from moneypenny.injection import InjectionQueue


def test_fifo_order():
    q = InjectionQueue()
    q.put("BRIEFING: A")
    q.put("CORRECTION: B")
    assert q.get() == "BRIEFING: A"
    assert q.get() == "CORRECTION: B"
    assert q.get() is None


def test_thread_safety_smoke():
    q = InjectionQueue()

    def producer():
        for i in range(100):
            q.put(f"BRIEFING: {i}")

    threads = [threading.Thread(target=producer) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    seen = 0
    while q.get() is not None:
        seen += 1
    assert seen == 400
