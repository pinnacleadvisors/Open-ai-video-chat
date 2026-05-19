import asyncio

import pytest

from server.utils.fanout import Broadcaster


@pytest.mark.asyncio
async def test_single_subscriber_receives_all():
    b: Broadcaster[int] = Broadcaster()
    sub = b.subscribe()
    await b.publish(1)
    await b.publish(2)
    await b.close()
    got = [x async for x in sub]
    assert got == [1, 2]


@pytest.mark.asyncio
async def test_multiple_subscribers_each_receive_all():
    b: Broadcaster[int] = Broadcaster()
    s1 = b.subscribe()
    s2 = b.subscribe()
    await b.publish("a")  # type: ignore[arg-type]
    await b.publish("b")  # type: ignore[arg-type]
    await b.close()
    got1 = [x async for x in s1]
    got2 = [x async for x in s2]
    assert got1 == got2 == ["a", "b"]


@pytest.mark.asyncio
async def test_slow_subscriber_drops_oldest():
    b: Broadcaster[int] = Broadcaster(maxsize=2)
    sub = b.subscribe()
    for i in range(5):
        await b.publish(i)
    await b.close()
    got = [x async for x in sub]
    # oldest items dropped — newest two preserved
    assert got[-2:] == [3, 4]
    assert len(got) <= 2


@pytest.mark.asyncio
async def test_subscriber_count_tracks_subscriptions():
    b: Broadcaster[int] = Broadcaster()
    assert b.subscriber_count == 0
    s1 = b.subscribe()
    s2 = b.subscribe()
    assert b.subscriber_count == 2
    await s1.aclose()
    assert b.subscriber_count == 1
    await s2.aclose()
    assert b.subscriber_count == 0


@pytest.mark.asyncio
async def test_publish_after_close_is_noop():
    b: Broadcaster[int] = Broadcaster()
    sub = b.subscribe()
    await b.close()
    await b.publish(1)
    got = [x async for x in sub]
    assert got == []


@pytest.mark.asyncio
async def test_consumers_run_concurrently():
    b: Broadcaster[int] = Broadcaster()
    s1 = b.subscribe()
    s2 = b.subscribe()

    async def consume(sub):
        return [x async for x in sub]

    consumers = asyncio.gather(consume(s1), consume(s2))
    for i in range(3):
        await b.publish(i)
    await b.close()
    a, c = await consumers
    assert a == c == [0, 1, 2]
