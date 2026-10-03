import asyncio
import time


def wait_for_ml_lookback(es, params):
    while True:
        response = es.ml.get_datafeed_stats(datafeed_id=params["datafeed-id"])
        if response["datafeeds"][0]["state"] == "stopped":
            break
        time.sleep(5)


async def wait_for_ml_lookback_async(es, params):
    while True:
        response = await es.ml.get_datafeed_stats(datafeed_id=params["datafeed-id"])
        if response["datafeeds"][0]["state"] == "stopped":
            break
        await asyncio.sleep(5)


# --- Experiment only (snapshot-aware autoscaling, elastic/elasticsearch-team#5291) --------------------------------
# Watches the index for an auto-reshard and changes the experiment-only snapshot floor settings from
# elastic/elasticsearch-serverless#7881 at the right moments. Modes:
#   none           record only
#   event-prescale set the floor and hold when the split starts, clear them after the catch-up snapshot succeeds
#   hold-only      set only the hold when the split starts, clear it after the catch-up
#   scaledown-mid  floor and hold are already set at project creation; drop them once the catch-up snapshot has been
#                  running for drop-after-seconds, so the autoscaler may remove nodes mid-upload
FLOOR = "serverless.autoscaling.indexing.snapshot_floor."


def _body(r):
    return r.body if hasattr(r, "body") else r


async def _put_floor(es, node_memory, total_memory, hold):
    s = {FLOOR + "node_memory": node_memory, FLOOR + "total_memory": total_memory, FLOOR + "hold_scale_down": hold}
    await es.cluster.put_settings(persistent=s)
    return s


async def snapshot_scaling_controller_async(es, params):
    import logging

    log = logging.getLogger(__name__)
    index, mode = params.get("index", "nyc_taxis"), params.get("mode", "none")
    poll, timeout = float(params.get("poll-interval", 15)), float(params.get("timeout", 10800))
    drop_after = float(params.get("drop-after-seconds", 60))
    events = []

    def ev(name, **kw):
        kw.update(event=name, t=int(time.time() * 1000))
        events.append(kw)
        log.info("snapshot-scaling-controller %s", kw)

    deadline = time.time() + timeout

    async def shards():
        try:
            r = _body(await es.indices.get_settings(index=index))
            return int(r[index]["settings"]["index"]["number_of_shards"])
        except Exception:
            return None

    initial = None
    while initial is None and time.time() < deadline:
        initial = await shards()
        if initial is None:
            await asyncio.sleep(poll)
    ev("start", mode=mode, initial_shards=initial)

    while time.time() < deadline:
        n = await shards()
        if n is not None and n > initial:
            break
        await asyncio.sleep(poll)
    split_ms = int(time.time() * 1000)
    ev("split_detected", shards=n)

    if mode == "event-prescale":
        ev("floor_set", settings=await _put_floor(es, params.get("node-memory"), params.get("total-memory"), True))
    elif mode == "hold-only":
        ev("floor_set", settings=await _put_floor(es, None, None, True))

    dropped = False
    while time.time() < deadline:
        r = _body(await es.perform_request(method="GET", path="/_snapshot/backup/_all", params={"sort": "start_time", "order": "desc", "size": "5"}))
        done = [s for s in r.get("snapshots", []) if s.get("start_time_in_millis", 0) > split_ms and s.get("state") == "SUCCESS"]
        if done:
            s = done[-1]
            ev("catchup_done", snapshot=s.get("snapshot"), start=s.get("start_time_in_millis"), end=s.get("end_time_in_millis"))
            break
        if mode == "scaledown-mid" and not dropped:
            cur = _body(await es.perform_request(method="GET", path="/_snapshot/backup/_current"))
            for s in cur.get("snapshots", []):
                st = s.get("start_time_in_millis", 0)
                if st > split_ms and time.time() * 1000 - st > drop_after * 1000:
                    ev("floor_dropped_mid_catchup", snapshot=s.get("snapshot"), running_s=int(time.time() - st / 1000),
                       settings=await _put_floor(es, None, None, None))
                    dropped = True
                    break
        await asyncio.sleep(poll)

    if mode != "none":
        ev("floor_cleared", settings=await _put_floor(es, None, None, None))
    ev("end")
    return {"weight": 1, "unit": "ops", "success": True, "events": events}


# --- Experiment only: oracle runs without a split (elastic/elasticsearch-team#5291) -------------------------------
# The index is created with a fixed shard count and SLM is stopped first, so nothing is snapshotted during ingest.
# Once ingest is quiet, one manual snapshot uploads the whole index, and the floor settings are changed around it.
# A background task logs every change in the set of index nodes and in per-shard snapshot progress, so scale-up lag,
# node removals and restarted shard uploads show up in the Rally log. Modes:
#   control      snapshot, change nothing
#   drop-nohold  drop the creation-time floor, wait drop-lead-seconds, snapshot (the 30 min scale-down lands mid-upload)
#   drop-hold    same, but set hold_scale_down just before the snapshot and clear it after
#   vstep        snapshot, then raise the floor step-after-seconds in (a resize in the middle of the upload)
#   prescale     raise the floor, wait for the new nodes, then snapshot
#   dual         snapshot to the main repository and to a second one (registered by prepare with dr-probe) at once
async def _index_nodes(es):
    r = _body(await es.perform_request(method="GET", path="/_nodes", params={"filter_path": "nodes.*.name,nodes.*.roles,nodes.*.attributes"}))
    out = {}
    for nid, n in r.get("nodes", {}).items():
        if "index" in n.get("roles", []):
            out[nid] = n.get("name", nid)
    return out


async def _await_snapshot(es, repo, name, deadline, poll, ev):
    while time.time() < deadline:
        r = _body(await es.perform_request(method="GET", path="/_snapshot/%s/%s" % (repo, name)))
        s0 = r.get("snapshots", [{}])[0]
        if s0.get("state") in ("SUCCESS", "PARTIAL", "FAILED"):
            ev("snapshot_done", repository=repo, state=s0.get("state"), start=s0.get("start_time_in_millis"), end=s0.get("end_time_in_millis"),
               duration_s=round((s0.get("end_time_in_millis", 0) - s0.get("start_time_in_millis", 0)) / 1000, 1),
               shards=s0.get("shards"), failures=s0.get("failures"))
            return
        await asyncio.sleep(poll)


async def snapshot_oracle_async(es, params):
    import logging

    log = logging.getLogger(__name__)
    index, mode, repo = params.get("index", "nyc_taxis"), params.get("mode", "control"), params.get("repository", "backup")
    poll = float(params.get("poll-interval", 10))
    quiet, settle = float(params.get("ingest-quiet-seconds", 180)), float(params.get("settle-seconds", 600))
    drop_lead, step_after = float(params.get("drop-lead-seconds", 1560)), float(params.get("step-after-seconds", 60))
    step_node, step_total = params.get("step-node-memory", "28gb"), params.get("step-total-memory", "56gb")
    deadline = time.time() + float(params.get("timeout", 14400))
    events = []

    def ev(name, **kw):
        kw.update(event=name, t=int(time.time() * 1000), ts=time.strftime("%H:%M:%S", time.gmtime()))
        events.append(kw)
        log.info("snapshot-oracle %s", kw)

    if mode == "prepare":
        for path in ("/_slm/stop",):
            try:
                ev("slm_stop", response=_body(await es.perform_request(method="POST", path=path)))
            except Exception as e:
                ev("slm_stop_failed", error=str(e)[:300])
        if params.get("dr-probe", False):
            try:
                b = _body(await es.perform_request(method="GET", path="/_snapshot/" + repo))[repo]
                st = dict(b.get("settings", {}))
                st["base_path"] = (st.get("base_path", "") + "-dr-probe").lstrip("-")
                await es.perform_request(method="PUT", path="/_snapshot/dr-probe", body={"type": b["type"], "settings": st})
                ev("dr_probe_ok", type=b["type"], keys=sorted(st.keys()))
            except Exception as e:
                ev("dr_probe_failed", error=str(e)[:300])
        return {"weight": 1, "unit": "ops", "success": True, "events": events}

    stop = asyncio.Event()

    async def watch_nodes():
        prev = None
        while not stop.is_set():
            try:
                cur = await _index_nodes(es)
                if cur != prev:
                    ev("index_nodes", nodes=sorted(cur.values()))
                    prev = cur
            except Exception as e:
                ev("index_nodes_error", error=str(e)[:200])
            await asyncio.sleep(poll)

    watcher = asyncio.ensure_future(watch_nodes())
    ev("start", mode=mode)

    # wait for ingest to go quiet
    last, since, errors = None, time.time(), 0
    while time.time() < deadline:
        try:
            c = _body(await es.perform_request(method="GET", path="/%s/_count" % index))["count"]
        except Exception as e:
            c = None
            errors += 1
            if errors <= 3:
                ev("count_error", error=repr(e)[:300])
        if c and c == last:
            if time.time() - since >= quiet:
                break
        else:
            last, since = c, time.time()
        await asyncio.sleep(30)
    ev("ingest_quiet", docs=last)

    if mode in ("drop-nohold", "drop-hold"):
        await asyncio.sleep(settle)
        ev("floor_dropped", settings=await _put_floor(es, None, None, None))
        await asyncio.sleep(drop_lead)
        if mode == "drop-hold":
            ev("hold_set", settings=await _put_floor(es, None, None, True))
    elif mode == "prescale":
        before = await _index_nodes(es)
        ev("floor_raised", settings=await _put_floor(es, step_node, step_total, None))
        stable_since, prev = None, None
        while time.time() < deadline:
            cur = await _index_nodes(es)
            if set(cur) & set(before):
                stable_since, prev = None, cur
            elif cur != prev:
                stable_since, prev = time.time(), cur
            elif time.time() - stable_since >= 60:
                break
            await asyncio.sleep(poll)
        ev("prescale_ready", nodes=sorted(prev.values()) if prev else None)
    else:
        await asyncio.sleep(120)

    name = "oracle-" + mode + "-" + time.strftime("%H%M%S", time.gmtime())
    targets = [(repo, name)] + ([(params.get("second-repository", "dr-probe"), name + "-dr")] if mode == "dual" else [])
    for r_, n_ in targets:
        await es.perform_request(method="PUT", path="/_snapshot/%s/%s" % (r_, n_), params={"wait_for_completion": "false"},
                                 body={"indices": index, "include_global_state": False})
        ev("snapshot_started", repository=r_, snapshot=n_)
    t0 = time.time()
    for r_, n_ in targets[1:]:
        asyncio.ensure_future(_await_snapshot(es, r_, n_, deadline, poll, ev))
    stepped, shards_prev, state = False, {}, None
    while time.time() < deadline:
        if mode == "vstep" and not stepped and time.time() - t0 >= step_after:
            ev("floor_raised_mid_snapshot", settings=await _put_floor(es, step_node, step_total, None))
            stepped = True
        try:
            st = _body(await es.perform_request(method="GET", path="/_snapshot/%s/%s/_status" % (repo, name)))
            snap = st["snapshots"][0]
            state = snap.get("state")
            cur = {}
            for sid, sh in snap.get("indices", {}).get(index, {}).get("shards", {}).items():
                cur[sid] = (sh.get("stage"), sh.get("node"), round(sh.get("stats", {}).get("processed", {}).get("size_in_bytes", 0) / 2**30, 2),
                            round(sh.get("stats", {}).get("total", {}).get("size_in_bytes", 0) / 2**30, 2))
            changed = {k: v for k, v in cur.items() if shards_prev.get(k, (None, None))[:2] != v[:2]}
            if changed or int(time.time() - t0) % 60 < poll:
                ev("snapshot_status", state=state, shards=cur)
            shards_prev = cur
        except Exception as e:
            ev("snapshot_status_error", error=str(e)[:200])
        r = _body(await es.perform_request(method="GET", path="/_snapshot/%s/%s" % (repo, name)))
        s0 = r.get("snapshots", [{}])[0]
        if s0.get("state") in ("SUCCESS", "PARTIAL", "FAILED"):
            state = s0.get("state")
            ev("snapshot_done", state=state, start=s0.get("start_time_in_millis"), end=s0.get("end_time_in_millis"),
               duration_s=round((s0.get("end_time_in_millis", 0) - s0.get("start_time_in_millis", 0)) / 1000, 1),
               shards=s0.get("shards"), failures=s0.get("failures"))
            break
        await asyncio.sleep(poll)

    if mode == "drop-hold":
        ev("hold_cleared", settings=await _put_floor(es, None, None, None))
    await asyncio.sleep(float(params.get("observe-after-seconds", 300)))
    stop.set()
    await watcher
    try:
        ev("slm_start", response=_body(await es.perform_request(method="POST", path="/_slm/start")))
    except Exception as e:
        ev("slm_start_failed", error=str(e)[:200])
    ev("end")
    return {"weight": 1, "unit": "ops", "success": True, "events": events}


def register(registry):
    async_runner = registry.meta_data.get("async_runner", False)
    if async_runner:
        registry.register_runner("wait-for-ml-lookback", wait_for_ml_lookback_async, async_runner=True)
        registry.register_runner("snapshot-scaling-controller", snapshot_scaling_controller_async, async_runner=True)
        registry.register_runner("snapshot-oracle", snapshot_oracle_async, async_runner=True)
    else:
        registry.register_runner("wait-for-ml-lookback", wait_for_ml_lookback)
