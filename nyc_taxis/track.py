import asyncio
import json
import os
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
        r = _body(
            await es.perform_request(
                method="GET", path="/_snapshot/backup/_all", params={"sort": "start_time", "order": "desc", "size": "5"}
            )
        )
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
                    ev(
                        "floor_dropped_mid_catchup",
                        snapshot=s.get("snapshot"),
                        running_s=int(time.time() - st / 1000),
                        settings=await _put_floor(es, None, None, None),
                    )
                    dropped = True
                    break
        await asyncio.sleep(poll)

    if mode != "none":
        ev("floor_cleared", settings=await _put_floor(es, None, None, None))
    ev("end")
    return {"weight": 1, "unit": "ops", "success": True}


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
    r = _body(
        await es.perform_request(method="GET", path="/_nodes", params={"filter_path": "nodes.*.name,nodes.*.roles,nodes.*.attributes"})
    )
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
            ev(
                "snapshot_done",
                repository=repo,
                state=s0.get("state"),
                start=s0.get("start_time_in_millis"),
                end=s0.get("end_time_in_millis"),
                duration_s=round((s0.get("end_time_in_millis", 0) - s0.get("start_time_in_millis", 0)) / 1000, 1),
                shards=s0.get("shards"),
                failures=s0.get("failures"),
            )
            return
        await asyncio.sleep(poll)


async def snapshot_oracle_async(es, params):
    import logging

    try:
        return await _snapshot_oracle(es, params)
    except BaseException as e:
        logging.getLogger(__name__).exception("snapshot-oracle {'event': 'runner_failed', 'error': %r}", repr(e)[:300])
        raise


async def _snapshot_oracle(es, params):
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
        names = (["dr-probe"] if params.get("dr-probe", False) else []) + [
            "dr-probe-%d" % i for i in range(1, int(params.get("extra-repositories", 0)) + 1)
        ]
        for rn in names:
            try:
                b = _body(await es.perform_request(method="GET", path="/_snapshot/" + repo))[repo]
                st = dict(b.get("settings", {}))
                st["base_path"] = (st.get("base_path", "") + "-" + rn).lstrip("-")
                await es.perform_request(method="PUT", path="/_snapshot/" + rn, body={"type": b["type"], "settings": st})
                ev("dr_probe_ok", repository=rn, type=b["type"], keys=sorted(st.keys()))
            except Exception as e:
                ev("dr_probe_failed", repository=rn, error=str(e)[:300])
        return {"weight": 1, "unit": "ops", "success": True}

    node_state = {"prev": None}

    async def tick():
        try:
            cur = await _index_nodes(es)
            if cur != node_state["prev"]:
                ev("index_nodes", nodes=sorted(cur.values()))
                node_state["prev"] = cur
        except Exception as e:
            ev("index_nodes_error", error=repr(e)[:200])

    async def nap(seconds):
        end_ = time.time() + seconds
        while time.time() < end_:
            await tick()
            await asyncio.sleep(min(poll, max(0.0, end_ - time.time())))

    ev("start", mode=mode)
    if params.get("initial-node-memory") or params.get("initial-total-memory"):
        ev("floor_initial", settings=await _put_floor(es, params.get("initial-node-memory"), params.get("initial-total-memory"), None))

    # wait for ingest to go quiet
    last, since, errors = None, time.time(), 0
    while time.time() < deadline:
        try:
            # index-tier stats; a _count would go to the search tier, which may not serve the index yet (503)
            st_ = _body(await es.perform_request(method="GET", path="/%s/_stats/indexing" % index))
            c = st_["_all"]["primaries"]["indexing"]["index_total"]
        except Exception as e:
            c = None
            errors += 1
            if errors <= 3:
                ev("count_error", error=repr(e)[:300])
        ev("ingest_count", docs=c, stable_s=int(time.time() - since))
        if c and c == last:
            if time.time() - since >= quiet:
                break
        else:
            last, since = c, time.time()
        await nap(float(params.get("count-interval", 30)))
    ev("ingest_quiet", docs=last)

    if mode in ("drop-nohold", "drop-hold"):
        await nap(settle)
        hold_from_drop = mode == "drop-hold" and bool(params.get("hold-from-drop", False))
        ev("floor_dropped", settings=await _put_floor(es, None, None, True if hold_from_drop else None))
        t_drop = time.time()
        await nap(drop_lead)
        if mode == "drop-hold" and not hold_from_drop:
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
            await tick()
            await asyncio.sleep(poll)
        ev("prescale_ready", nodes=sorted(prev.values()) if prev else None)
    else:
        await nap(float(params.get("pre-snapshot-seconds", 120)))

    name = "oracle-" + mode + "-" + time.strftime("%H%M%S", time.gmtime())
    targets = [(repo, name)] + ([(params.get("second-repository", "dr-probe"), name + "-dr")] if mode == "dual" else [])
    targets += [("dr-probe-%d" % i, name + "-dr%d" % i) for i in range(1, int(params.get("extra-repositories", 0)) + 1)]
    for r_, n_ in targets:
        await es.perform_request(
            method="PUT",
            path="/_snapshot/%s/%s" % (r_, n_),
            params={"wait_for_completion": "false"},
            body={"indices": index, "include_global_state": False},
        )
        ev("snapshot_started", repository=r_, snapshot=n_)
    t0 = time.time()
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
                cur[sid] = (
                    sh.get("stage"),
                    sh.get("node"),
                    round(sh.get("stats", {}).get("processed", {}).get("size_in_bytes", 0) / 2**30, 2),
                    round(sh.get("stats", {}).get("total", {}).get("size_in_bytes", 0) / 2**30, 2),
                )
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
            ev(
                "snapshot_done",
                state=state,
                start=s0.get("start_time_in_millis"),
                end=s0.get("end_time_in_millis"),
                duration_s=round((s0.get("end_time_in_millis", 0) - s0.get("start_time_in_millis", 0)) / 1000, 1),
                shards=s0.get("shards"),
                failures=s0.get("failures"),
            )
            break
        await tick()
        await asyncio.sleep(poll)

    for r_, n_ in targets[1:]:
        await _await_snapshot(es, r_, n_, deadline, poll, ev)
    if mode == "drop-hold":
        hold_until = float(params.get("hold-until-seconds-after-drop", 0))
        if hold_until:
            await nap(max(0.0, t_drop + hold_until - time.time()))
        ev("hold_cleared", settings=await _put_floor(es, None, None, None))
    await nap(float(params.get("observe-after-seconds", 300)))
    try:
        ev("slm_start", response=_body(await es.perform_request(method="POST", path="/_slm/start")))
    except Exception as e:
        ev("slm_start_failed", error=str(e)[:200])
    ev("end")
    return {"weight": 1, "unit": "ops", "success": True}


# qos-baseline: QA baseline for snapshot network QoS (experiment only, not for merge).
# Modes: prepare (stop SLM, wait for the index tier, restore smoke test), seed (snapshot the seed indices, then restore
# copies so the first SLM snapshot has a catch-up of at least catchup-target-seconds), phase-b (start SLM, run the
# policy once, restore one index under load as phase C, record every snapshot). Every step is logged as an event.
def _shard_summary(status):
    times, sizes = [], []
    for idx in status.get("indices", {}).values():
        for sh in idx.get("shards", {}).values():
            st = sh.get("stats", {})
            times.append(st.get("time_in_millis", 0) / 1000)
            sizes.append(st.get("incremental", {}).get("size_in_bytes", 0))
    times.sort()
    pick = lambda q: round(times[min(len(times) - 1, int(q * len(times)))], 1) if times else None
    return {
        "shards": len(times),
        "shard_s_p50": pick(0.5),
        "shard_s_p90": pick(0.9),
        "shard_s_max": pick(1.0),
        "incremental_gib": round(sum(sizes) / 2**30, 2),
        "per_shard_s": [round(t, 1) for t in times],
    }


# ----------------------------------------------------------------------------------------------------------------------
# m2s1 (shard splits): helpers for the qos-baseline modes "splits-seed" and "splits-run". Kept at module level so that the fake-ES tests can import them.
# ----------------------------------------------------------------------------------------------------------------------
GIB = 2**30
CONTROL_PATH = "~/m2s1-control.json"


def _control():
    """Operator overrides read at every loop iteration of the splits run: {"threshold_gib": n, "detect": "state"|"shards-stable", "feed_pause": bool, "expunge": bool}."""
    try:
        with open(os.path.expanduser(CONTROL_PATH)) as f:
            return json.load(f)
    except Exception:
        return {}


class _Feeder:
    """Writes documents of the nyc_taxis corpus (one JSON document per line) into a single index through its own bulk requests, so that the run can pause and resume
    the writes of that index at will (a Rally bulk task cannot be paused). pause() returns only when no request is in flight."""

    def __init__(self, req, ev, index, path, batch=5000, concurrency=8, offset_fraction=0.5):
        self.req, self.ev, self.index, self.path = req, ev, index, os.path.expanduser(path)
        self.batch, self.concurrency = batch, concurrency
        self.paused, self.stopped, self.in_flight = True, False, 0
        self.docs, self.rejections, self.errors, self.bulk_errors = 0, 0, 0, 0
        self._lock = asyncio.Lock()
        self._fh = None
        self._offset_fraction = offset_fraction
        self._tasks = []

    def _open(self):
        self._fh = open(self.path, "rb")
        size = os.fstat(self._fh.fileno()).st_size
        self._fh.seek(int(size * self._offset_fraction))
        self._fh.readline()  # align to the next line start

    def _read_batch(self):
        lines = []
        while len(lines) < self.batch:
            line = self._fh.readline()
            if not line:
                self._fh.seek(0)
                continue
            lines.append(line if line.endswith(b"\n") else line + b"\n")
        return b"".join(b'{"index":{}}\n' + l for l in lines), len(lines)

    async def _worker(self):
        while not self.stopped:
            if self.paused:
                await asyncio.sleep(1)
                continue
            self.in_flight += 1   # claimed before the batch is read: pause() waits for every claimed worker, and a worker that claimed before the pause re-checks it before sending
            try:
                async with self._lock:
                    if self._fh is None:
                        await asyncio.to_thread(self._open)
                    body, n = await asyncio.to_thread(self._read_batch)
                for attempt in range(8):
                    if self.paused:
                        break
                    try:
                        r = await self.req(
                            "POST",
                            "/%s/_bulk" % self.index,
                            body=body,
                            params_={"filter_path": "errors,items.*.status"},
                            headers={"Content-Type": "application/x-ndjson", "Accept": "application/json"},
                        )
                    except Exception as e:
                        st = getattr(getattr(e, "meta", None), "status", 0) or getattr(e, "status_code", 0) or 0
                        if st == 429:
                            self.rejections += 1
                        else:
                            self.errors += 1
                        await asyncio.sleep(min(30, 2 * (attempt + 1)))
                        continue
                    if r.get("errors"):
                        self.bulk_errors += 1
                        bad = sum(1 for it in r.get("items", []) if list(it.values())[0].get("status", 200) >= 300)
                        if bad == n:
                            await asyncio.sleep(min(30, 2 * (attempt + 1)))
                            continue
                    self.docs += n
                    break
            finally:
                self.in_flight -= 1

    def start(self):
        self._tasks = [asyncio.ensure_future(self._worker()) for _ in range(self.concurrency)]

    async def resume(self, why):
        if self.paused:
            self.paused = False
            self.ev("feed_resumed", why=why, docs=self.docs)

    async def pause(self, why):
        was = self.paused
        self.paused = True
        while self.in_flight:
            await asyncio.sleep(0.5)
        if not was:
            self.ev("feed_paused", why=why, docs=self.docs, rejections=self.rejections, errors=self.errors, bulk_errors=self.bulk_errors)

    async def stop(self):
        await self.pause("stop")
        self.stopped = True
        for t in self._tasks:
            t.cancel()


async def _index_usage(req, index):
    """Per primary shard of the index: total data set bytes (what the auto-reshard monitor compares), docs and deleted docs, from indices stats; plus the number of primary shards
    in the index settings. -> {"shards": {sid: {"bytes", "docs", "deleted"}}, "n_shards": int, "total", "avg", "deletes_pct"}"""
    st = await req("GET", "/%s/_stats/store,docs" % index, params_={"level": "shards", "filter_path": "indices.*.shards.*.routing.primary,indices.*.shards.*.store,indices.*.shards.*.docs"}, retry=True)
    shards = {}
    for _, v in (st.get("indices") or {}).items():
        for sid, copies in (v.get("shards") or {}).items():
            for c in copies:
                if not (c.get("routing") or {}).get("primary", True):
                    continue
                s = c.get("store") or {}
                d = c.get("docs") or {}
                cur = shards.setdefault(int(sid), {"bytes": 0, "docs": 0, "deleted": 0})
                cur["bytes"] = max(cur["bytes"], s.get("total_data_set_size_in_bytes", s.get("size_in_bytes", 0)))
                cur["docs"] = max(cur["docs"], d.get("count", 0))
                cur["deleted"] = max(cur["deleted"], d.get("deleted", 0))
    se = await req("GET", "/%s/_settings" % index, params_={"filter_path": "*.settings.index.number_of_shards"}, retry=True)
    n = int(next(iter(se.values()))["settings"]["index"]["number_of_shards"]) if se else len(shards)
    total = sum(s["bytes"] for s in shards.values())
    docs = sum(s["docs"] for s in shards.values())
    dele = sum(s["deleted"] for s in shards.values())
    return {"shards": shards, "n_shards": n, "total": total, "avg": total // max(1, n), "deletes_pct": round(100.0 * dele / (docs + dele), 2) if docs + dele else 0.0}


async def _resharding_state(req, index, mode):
    """True if resharding metadata exists for the index (a split is running), False if not, None if it cannot be read.
    mode "state": cluster state metadata of the index (key "resharding", IndexMetadata.KEY_RESHARDING); the index has to be listed (its "state" is requested too), so an
    empty answer from a wrong key or a filtered route is "cannot be read", never "no resharding". Any other mode: None (the caller decides)."""
    if mode != "state":
        return None
    try:
        r = await req(
            "GET",
            "/_cluster/state/metadata/%s" % index,
            params_={"filter_path": "metadata.indices.%s.state,metadata.indices.%s.resharding" % (index, index)},
        )
    except Exception:
        return None
    imd = ((r.get("metadata") or {}).get("indices") or {}).get(index)
    if not imd or "state" not in imd:
        return None
    return bool(imd.get("resharding"))


async def qos_baseline_async(es, params):
    import logging

    try:
        return await _qos_baseline(es, params)
    except BaseException as e:
        logging.getLogger(__name__).exception("qos-baseline {'event': 'runner_failed', 'error': %r}", repr(e)[:300])
        raise


async def _qos_baseline(es, params):
    import logging
    import math

    log = logging.getLogger(__name__)
    mode, repo = params.get("mode"), params.get("repository", "backup")
    prefix, n_idx = params.get("prefix", "qos-"), int(params.get("indices", 4))
    seeds = ["%s%d" % (prefix, i) for i in range(n_idx)]
    poll = float(params.get("poll-interval", 10))
    deadline = time.time() + float(params.get("timeout", 14400))

    def ev(name, **kw):
        kw.update(event=name, t=int(time.time() * 1000), ts=time.strftime("%H:%M:%S", time.gmtime()))
        log.info("qos-baseline %s", kw)

    def status_of(e):
        return getattr(getattr(e, "meta", None), "status", 0) or getattr(e, "status_code", 0) or 0

    def transient(e, any_5xx=False):
        st = status_of(e)
        return (
            type(e).__name__ in ("ConnectionTimeout", "ConnectionError", "TransportError")
            or st in (429, 502, 503, 504)
            or (any_5xx and st >= 500)
        )

    async def req(method, path, body=None, params_=None, retry=False, headers=None):
        # reads and deletes survive a slow or briefly unreachable endpoint. Writes are retried only when the caller says the call is
        # idempotent (retry=True): the settings PUT, _flush, _slm/start. Restores and snapshot creation have their own handling.
        attempts = 6 if (method in ("GET", "DELETE") or retry) else 1
        for i in range(attempts):
            try:
                return _body(await es.perform_request(method=method, path=path, body=body, params=params_, **({"headers": headers} if headers else {})))
            except Exception as e:
                if not transient(e, any_5xx=retry) or i == attempts - 1:
                    raise
                ev("request_retry", method=method, path=path[:80], attempt=i + 1, error=type(e).__name__, status=status_of(e))
                await asyncio.sleep(min(30, 5 * (i + 1)))

    def save_status(name, s0, snap_status):
        # per-shard incremental sizes of a finished snapshot, written once here (the status call is already made) so that the measurement can read
        # a small file instead of asking the cluster for the status of 300+ shards again
        try:
            shards = {}
            for idx, v in (snap_status.get("indices") or {}).items():
                for sid, sh in (v.get("shards") or {}).items():
                    shards["%s|%s" % (idx, sid)] = ((sh.get("stats") or {}).get("incremental") or {}).get("size_in_bytes", 0)
            stats = snap_status.get("stats") or {}
            path = os.path.expanduser("~/qos-status-%s.json" % name)
            with open(path, "w") as f:
                json.dump(
                    {
                        "snapshot": name,
                        "state": s0.get("state"),
                        "start": s0.get("start_time_in_millis"),
                        "end": s0.get("end_time_in_millis"),
                        "total_bytes": (stats.get("total") or {}).get("size_in_bytes"),
                        "incremental_bytes": (stats.get("incremental") or {}).get("size_in_bytes"),
                        "time_ms": stats.get("time_in_millis"),
                        "shards": shards,
                    },
                    f,
                )
        except Exception as e:  # never let bookkeeping stop the run
            ev("save_status_failed", snapshot=name, error=repr(e)[:200])

    async def wait_snapshot(name):
        while time.time() < deadline:
            s0 = (await req("GET", "/_snapshot/%s/%s" % (repo, name))).get("snapshots", [{}])[0]
            if s0.get("state") in ("SUCCESS", "PARTIAL", "FAILED"):
                st = await req("GET", "/_snapshot/%s/%s/_status" % (repo, name))
                tot = st.get("snapshots", [{}])[0].get("stats", {})
                save_status(name, s0, st.get("snapshots", [{}])[0])
                ev(
                    "snapshot_done",
                    snapshot=name,
                    state=s0.get("state"),
                    duration_s=round((s0.get("end_time_in_millis", 0) - s0.get("start_time_in_millis", 0)) / 1000, 1),
                    start=s0.get("start_time_in_millis"),
                    end=s0.get("end_time_in_millis"),
                    failures=s0.get("failures"),
                    total_gib=round(tot.get("total", {}).get("size_in_bytes", 0) / 2**30, 2),
                    **_shard_summary(st.get("snapshots", [{}])[0]),
                )
                return s0, tot
            await asyncio.sleep(poll)
        raise TimeoutError("snapshot %s" % name)

    async def wait_green(indices):
        errors = 0
        while time.time() < deadline:
            try:
                h = await req("GET", "/_cluster/health/" + indices, params_={"wait_for_status": "green", "timeout": "20s"})
                if h.get("status") == "green":
                    return
            except Exception as e:
                # a slow or busy endpoint must not end the run; keep polling until the deadline
                errors += 1
                if errors <= 3 or errors % 20 == 0:
                    ev("health_error", indices=indices, errors=errors, error=repr(e)[:200])
            await asyncio.sleep(poll)
        raise TimeoutError("green " + indices)

    async def restore(snapshot, indices, rename_to):
        t0 = time.time()
        body = {
            "indices": indices,
            "include_global_state": False,
            "rename_pattern": "%s(.+)" % prefix,
            "rename_replacement": rename_to + "$1",
            "index_settings": {"index.number_of_replicas": 0},
        }
        path = "/_snapshot/%s/%s/_restore" % (repo, snapshot)

        async def exists():
            rows = await req("GET", "/_cat/indices/%s*" % rename_to, params_={"format": "json", "h": "index"})
            return bool(rows)

        async def appeared(seconds=60):
            # a timed-out POST may still be running on the master: give it time to show up before deciding it did not happen
            for _ in range(max(1, int(seconds // 5))):
                if await exists():
                    return True
                await asyncio.sleep(5)
            return await exists()

        async def post(b):
            # a restore is not idempotent: after a timeout, wait and look whether the request took effect before sending it again
            timed_out = False
            for i in range(6):
                try:
                    await req("POST", path, body=b)
                    return
                except Exception as e:
                    if not transient(e):
                        if timed_out and await exists():
                            # the earlier attempt took effect; this error (typically 'index already exists') is its echo
                            ev("restore_accepted_after_retry_error", rename_to=rename_to, status=status_of(e))
                            return
                        raise
                    timed_out = True
                    if i == 5:
                        raise
                    if await appeared():
                        ev("restore_accepted_after_timeout", rename_to=rename_to, error=type(e).__name__)
                        return
                    ev("restore_retry", rename_to=rename_to, attempt=i + 1, error=type(e).__name__, status=status_of(e))
                    await asyncio.sleep(min(30, 5 * (i + 1)))

        replicas = 0
        try:
            await post(body)
        except Exception as e:
            text = str(getattr(e, "body", "") or repr(e))
            # a replicas refusal: the body says so, or the operator is forbidden (403). Any other 400 is a different problem and is reported as it is
            refused = status_of(e) == 403 or "number_of_replicas" in text
            if not refused:
                if params.get("strict", False) and status_of(e) == 400:
                    invalid("restore_rejected", status=400, detail=text[:300])
                raise
            ev("restore_replicas0_refused", error=text[:500])
            if params.get("strict", False):
                # every arm must restore the same way: a restore with replicas would put copies on the search tier
                invalid("restore_replicas0_refused", error=text[:200])
            del body["index_settings"]
            await post(body)
            replicas = None
        ev("restore_start", snapshot=snapshot, indices=indices, rename_to=rename_to, replicas=replicas)
        await wait_green(rename_to + "*")
        ev("restore_done", snapshot=snapshot, rename_to=rename_to, duration_s=round(time.time() - t0, 1))

    SWITCH_KEYS = ("background_qos.enabled", "adaptive_upload_concurrency.enabled", "backlog_tracking.enabled")

    async def cluster_settings():
        r = await req("GET", "/_cluster/settings", params_={"flat_settings": "true", "filter_path": "persistent.*,transient.*"})
        return {
            k: v
            for tier in ("persistent", "transient")
            for k, v in (r.get(tier) or {}).items()
            if not k.startswith("serverless.autoscaling")
        }

    def switches_on(settings):
        return {k: v for k, v in settings.items() if any(k.endswith(x) for x in SWITCH_KEYS) and str(v).lower() == "true"}

    def invalid(reason, **kw):
        # a validity gate failed: the arm's data must not be compared. Stop the run, loudly.
        ev("arm_invalid", reason=reason, **kw)
        raise RuntimeError("arm invalid: %s %s" % (reason, kw))

    async def apply_arm_settings():
        want = params.get("apply-settings") or {}
        if not want:
            ev("settings_applied", applied={}, state=await cluster_settings())
            return
        await req("PUT", "/_cluster/settings", body={"persistent": want}, retry=True)
        state = await cluster_settings()
        missing = {k: v for k, v in want.items() if str(state.get(k)).lower() != str(v).lower()}
        ev("settings_applied", applied=want, state=state, missing=missing)
        if missing:
            invalid("settings_not_in_effect", missing=missing)

    async def write_shard_map(label):
        # which node holds each primary, so snapshot status per shard can be mapped to the node whose tracker reported it
        rows = await req("GET", "/_cat/shards", params_={"format": "json", "h": "index,shard,prirep,state,node,store"})
        path = os.path.expanduser("~/qos-shardmap-%s-%d.json" % (label, int(time.time())))
        with open(path, "w") as f:
            json.dump([r for r in rows if r.get("prirep") == "p"], f)
        ev("shard_map", path=path, primaries=sum(1 for r in rows if r.get("prirep") == "p"))

    async def throughput_probe(state):
        # achieved indexing over the live indices (primaries) since the last probe; the caller decides what to do with it
        try:
            st = await req("GET", "/%s/_stats/indexing" % ",".join(seeds))
            n, now = st["_all"]["primaries"]["indexing"]["index_total"], time.time()
        except Exception:
            return None
        last = state.get("last")
        state["last"] = (n, now)
        if not last or now - last[1] < 5:
            return None
        rate = (n - last[0]) / (now - last[1])
        target = float(params.get("target-docs-per-s", 0))
        ev("throughput_probe", docs_per_s=round(rate), pct_of_target=round(100 * rate / target, 1) if target else None)
        return rate

    async def plan_copies():
        # the seed snapshot runs with no foreground load: its rate is the arm's clean rate. Wall time includes
        # start-up and finalization, so the rate is low and the copy count errs long.
        st = (await req("GET", "/_snapshot/%s/seed/_status" % repo)).get("snapshots", [{}])[0].get("stats", {})
        seed_bytes = st.get("total", {}).get("size_in_bytes", 0)
        rate = seed_bytes / max(1.0, st.get("time_in_millis", 0) / 1000)
        target = float(params.get("catchup-target-seconds", 1200))
        copies = min(int(params.get("max-copies", 12)), max(1, math.ceil(target * rate / max(1, seed_bytes))))
        fixed = int(params.get("fixed-copies", 0))
        if fixed:
            # iteration 3: the same K for every arm of a size, whatever the arm's own seed rate was
            copies = fixed
        ev(
            "copies_planned",
            fixed=bool(fixed),
            seed_gib=round(seed_bytes / 2**30, 2),
            clean_rate_mib_s=round(rate / 2**20, 1),
            copies=copies,
            expected_catchup_s=round(copies * seed_bytes / max(1.0, rate)),
        )
        return copies

    def shape_ok(cur):
        """The index tier is on its target shape: an expected node count (a comma list is accepted, e.g. "2,3" for the 16 GiB arms), all nodes the same
        size, and, when expected-index-gib is given, that size (a comma list is accepted). Without the size a tier of the wrong step would pass.
        """
        counts = {int(x) for x in str(params.get("expected-index-nodes", 3)).split(",") if x.strip()}
        gibs = [float(x) for x in str(params.get("expected-index-gib", "")).split(",") if x.strip()]
        if len(cur) not in counts or len({n[1] for n in cur}) != 1:
            return False
        return not gibs or any(abs(cur[0][1] - g) <= 0.6 for g in gibs)

    async def index_nodes():
        r = await req(
            "GET",
            "/_nodes/stats/os,jvm",
            params_={"filter_path": "nodes.*.name,nodes.*.roles,nodes.*.os.mem.total_in_bytes,nodes.*.jvm.mem.heap_max_in_bytes"},
        )
        return sorted(
            (
                n["name"],
                round(n.get("os", {}).get("mem", {}).get("total_in_bytes", 0) / 2**30, 1),
                round(n.get("jvm", {}).get("mem", {}).get("heap_max_in_bytes", 0) / 2**30, 1),
            )
            for n in r.get("nodes", {}).values()
            if "index" in n.get("roles", [])
        )

    ev("start", mode=mode)
    if mode == "prepare":
        ev("slm_stop", response=await req("POST", "/_slm/stop"))
        # a rerun starts clean: drop copies and restores left by an earlier attempt, and the seed snapshot
        left = await req("GET", "/_cat/indices/%s*" % prefix, params_={"format": "json", "h": "index"})
        stale = [i["index"] for i in left if i["index"] not in seeds]
        for i in stale:
            await req("DELETE", "/" + i)
        try:
            await req("DELETE", "/_snapshot/%s/seed" % repo)
            stale.append("snapshot seed")
        except Exception:
            pass
        ev("cleanup", removed=stale)
        ev(
            "slm_policies",
            policies={
                k: {"schedule": v.get("policy", {}).get("schedule"), "repository": v.get("policy", {}).get("repository")}
                for k, v in (await req("GET", "/_slm/policy")).items()
            },
        )
        # wait until the index tier has the expected node count for stable-seconds (the floor needs a few minutes)
        stable = float(params.get("stable-seconds", 300))
        prev, since = None, time.time()
        while time.time() < deadline:
            cur = await index_nodes()
            if cur != prev:
                ev("index_nodes", nodes=cur)
                prev, since = cur, time.time()
            if shape_ok(cur) and time.time() - since >= stable:
                break
            await asyncio.sleep(30)
        # restore smoke test, so a refused restore fails the run in minutes rather than hours
        smoke = prefix + "smoke"
        for i in (smoke, prefix + "r0-smoke"):
            await req("DELETE", "/" + i, params_={"ignore_unavailable": "true"})
        await req("PUT", "/" + smoke)
        await req("POST", "/%s/_doc" % smoke, body={"ok": 1}, params_={"refresh": "true"})
        await req("PUT", "/_snapshot/%s/%s" % (repo, smoke), body={"indices": smoke, "include_global_state": False})
        await wait_snapshot(smoke)
        await restore(smoke, smoke, prefix + "r0-")
        ev("smoke_ok")
        for i in (smoke, prefix + "r0-smoke"):
            await req("DELETE", "/" + i)
        await req("DELETE", "/_snapshot/%s/%s" % (repo, smoke))

    elif mode == "seed":
        # the seed snapshot gives the arm's clean rate, so it must run on the target shape: wait until the index tier
        # has the expected node count, all the same size, for a while; give up after gate-seconds (relaunch the arm)
        stable, gate_end = (
            float(params.get("stable-seconds", 120)),
            time.time() + float(params.get("gate-seconds", 2700)),
        )
        prev, since = None, time.time()
        while True:
            cur = await index_nodes()
            if cur != prev:
                ev("index_nodes", nodes=cur)
                prev, since = cur, time.time()
            if shape_ok(cur) and time.time() - since >= stable:
                ev("seed_gate_open", nodes=cur)
                break
            if time.time() > gate_end:
                ev("seed_gate_timeout", nodes=cur)
                raise RuntimeError("index tier not on its target shape: %s" % (cur,))
            await asyncio.sleep(30)
        state = await cluster_settings()
        ev("settings_state", phase="seed", state=state)
        if params.get("strict", False) and switches_on(state):
            invalid("switches_on_at_seed", switches=switches_on(state))
        await req("POST", "/%s/_flush" % ",".join(seeds), retry=True)
        strict = params.get("strict", False)
        attempts = 2 if strict else int(params.get("seed-attempts", 3))
        reasons = []
        for attempt in range(1, attempts + 1):
            await req("PUT", "/_snapshot/%s/seed" % repo, body={"indices": ",".join(seeds), "include_global_state": False})
            s0, _ = await wait_snapshot("seed")
            if s0.get("state") == "SUCCESS":
                if attempt > 1:
                    # iteration 3: exactly one retry is allowed; the starting state after a successful retry is identical. Flagged, and the
                    # seed rate of a retried arm is informational only
                    ev("seed_retried", attempts=attempt, previous_failures=reasons)
                break
            # copies are restored from the seed, so it must be complete; the failed attempt stays in the log
            reasons.append(
                {
                    "attempt": attempt,
                    "state": s0.get("state"),
                    "failures": len(s0.get("failures") or []),
                    "reasons": sorted({(f.get("reason") or "")[:110] for f in (s0.get("failures") or [])})[:3],
                }
            )
            ev("seed_retry", attempt=attempt, state=s0.get("state"), reasons=reasons[-1]["reasons"])
            if attempt == attempts:
                if strict:
                    invalid("seed_not_success_after_retry", attempts=attempt, failures=reasons)
                break
            # nothing may still be running before the seed is taken again, and the failed seed must not be restorable
            while (await req("GET", "/_snapshot/%s/_current" % repo)).get("snapshots"):
                await asyncio.sleep(poll)
            try:
                await req("DELETE", "/_snapshot/%s/seed" % repo)
            except Exception as e:
                if status_of(e) != 404:  # a retried DELETE that finds it already gone is fine
                    raise
            # the delete runs on the master and the snapshot name stays taken until it is done: wait for the 404
            for _ in range(60):
                try:
                    await req("GET", "/_snapshot/%s/seed" % repo)
                except Exception as e:
                    if status_of(e) == 404:
                        break
                    raise
                await asyncio.sleep(5)
            else:
                invalid("seed_delete_not_finished")
        else:
            raise RuntimeError("seed snapshot not SUCCESS after retries")
        await plan_copies()

    elif mode == "phase-b":
        # warm-up under steady load, then phase C: restore K copies of the seed (the backlog), then start SLM and
        # run the policy once (the catch-up), then record every snapshot until the end. resume=true reruns this on
        # existing data (earlier copies stay, new ones get copy-tag d), after a crash of an earlier phase B.
        total, warmup = float(params.get("duration-seconds", 7200)), float(params.get("warmup-seconds", 600))
        tag = params.get("copy-tag", "c")
        t_end = time.time() + total
        policy = params.get("policy") or next(iter(await req("GET", "/_slm/policy")))
        monitor = bool(params.get("monitor-only", False))
        if not monitor:
            ev("slm_stop", response=await req("POST", "/_slm/stop"))
            while (await req("GET", "/_snapshot/%s/_current" % repo)).get("snapshots"):
                await asyncio.sleep(poll)

        async def policy_names():
            p = (await req("GET", "/_slm/policy/" + policy)).get(policy, {})
            names = {
                (p.get("last_success") or {}).get("snapshot_name"),
                (p.get("last_failure") or {}).get("snapshot_name"),
                (p.get("in_progress") or {}).get("name"),
            }
            return {n for n in names if n}

        before = await policy_names()
        first = None
        probe_state = {}
        if monitor:
            # an earlier Rally already ran the catch-up; only keep recording snapshots under the same steady load
            ev("monitor_only", already_known=sorted(before))
        else:
            warm_end, probe_state, rates = time.time() + warmup, {}, []
            while time.time() < warm_end:
                r = await throughput_probe(probe_state)
                if r is not None:
                    rates.append(r)
                await asyncio.sleep(min(60, max(0.0, warm_end - time.time())))
            target = float(params.get("target-docs-per-s", 0))
            if target and rates:
                tail = sorted(rates[-5:])
                ev("warm_gate", median_docs_per_s=round(tail[len(tail) // 2]), target=target, ok=tail[len(tail) // 2] >= 0.98 * target)
            copies, t0 = await plan_copies(), time.time()
            await apply_arm_settings()
            ev("phase_c_start", copies=copies, tag=tag)
            for k in range(1, copies + 1):
                await restore("seed", ",".join(seeds), "%s%s%d-" % (prefix, tag, k))
            ev("phase_c_done", copies=copies, duration_s=round(time.time() - t0, 1))
            # let the tracker finish reading the restored copies before the catch-up, the same for every arm
            await asyncio.sleep(float(params.get("pre-catchup-settle-seconds", 0)))
            await write_shard_map("precatchup")
            ev("settings_state", phase="pre_catchup", state=await cluster_settings())
            ev("slm_start", response=await req("POST", "/_slm/start", retry=True), policy=policy)
            known = set(await policy_names())
            try:
                first = (await req("POST", "/_slm/policy/%s/_execute" % policy)).get("snapshot_name")
            except Exception as e:
                # the execute call is not idempotent: do not repeat it. If it started the snapshot, take the name from the policy
                ev("execute_failed", error=type(e).__name__, status=status_of(e))
                first = None
                for _ in range(12):
                    p = (await req("GET", "/_slm/policy/" + policy)).get(policy, {})
                    first = (p.get("in_progress") or {}).get("name") or next(iter(sorted(await policy_names() - known)), None)
                    if first:
                        break
                    await asyncio.sleep(5)
                if not first:
                    invalid("catchup_not_started", error=type(e).__name__)
            ev("catchup_started", snapshot=first)
        seen = set(before)
        errors = 0
        while time.time() < t_end:
            await throughput_probe(probe_state if not monitor else {})
            try:
                for n in sorted(await policy_names() - seen):
                    if n in ("seed",):
                        continue
                    cur = (await req("GET", "/_snapshot/%s/%s" % (repo, n))).get("snapshots", [{}])[0]
                    if cur.get("state") in ("SUCCESS", "PARTIAL", "FAILED"):
                        seen.add(n)
                        await wait_snapshot(n)
            except Exception as e:
                errors += 1
                ev("loop_error", errors=errors, error=repr(e)[:200])
            await asyncio.sleep(30)
        ev("phase_b_done", snapshots=sorted(seen - before), catchup=first, loop_errors=errors)
    elif mode == "splits-seed":
        # as the seed mode up to the seed itself: the index tier is on its target shape and the switches are off, then flush. No seed snapshot and no restores: the
        # arm does not catch up; the seed indices only have to exist at a known shard size (the threshold is derived from it in splits-run).
        stable, gate_end = float(params.get("stable-seconds", 120)), time.time() + float(params.get("gate-seconds", 2700))
        prev, since = None, time.time()
        while True:
            cur = await index_nodes()
            if cur != prev:
                ev("index_nodes", nodes=cur)
                prev, since = cur, time.time()
            if shape_ok(cur) and time.time() - since >= stable:
                ev("seed_gate_open", nodes=cur)
                break
            if time.time() > gate_end:
                ev("seed_gate_timeout", nodes=cur)
                raise RuntimeError("index tier not on its target shape: %s" % (cur,))
            await asyncio.sleep(30)
        state = await cluster_settings()
        ev("settings_state", phase="seed", state=state)
        if params.get("strict", False) and switches_on(state):
            invalid("switches_on_at_seed", switches=switches_on(state))
        await req("POST", "/%s/_flush" % ",".join(seeds), retry=True)
        sizes = {}
        for s_ in seeds:
            u = await _index_usage(req, s_)
            sizes[s_] = {"n_shards": u["n_shards"], "max_shard_bytes": max([v["bytes"] for v in u["shards"].values()] or [0]), "total_bytes": u["total"]}
        ev("splits_seed_done", sizes=sizes, max_shard_gib=round(max(v["max_shard_bytes"] for v in sizes.values()) / GIB, 2))

    elif mode == "splits-run":
        # warm-up under the steady load, then: settings on, threshold T from the seed shard sizes, a pre-split snapshot of the split index, split 1 (1 -> 2), its snapshot,
        # split 2 (2 -> 4), its snapshot. SLM stays stopped for the whole run: every snapshot is a direct PUT.
        split_idx = params.get("split-index", "split-1p")
        total_s, warmup = float(params.get("duration-seconds", 21600)), float(params.get("warmup-seconds", 600))
        t_end = time.time() + total_s
        policy = params.get("policy") or next(iter(await req("GET", "/_slm/policy")))
        ev("slm_stop", response=await req("POST", "/_slm/stop"))
        while (await req("GET", "/_snapshot/%s/_current" % repo)).get("snapshots"):
            await asyncio.sleep(poll)
        settle_s = float(params.get("tracker-settle-seconds", 150))
        trigger_timeout = float(params.get("split-trigger-timeout-seconds", 2400))
        corpus = params.get("corpus-path", "~/.rally/benchmarks/data/qos/documents.json")
        names = []

        def attention(reason, **kw):
            ev("splits_attention", reason=reason, **kw)

        ticks = {}

        def every(key, secs):
            if time.time() - ticks.get(key, 0) >= secs:
                ticks[key] = time.time()
                return True
            return False

        # warm window: the seed indices grow under the foreground load; their growth sets the threshold
        probe_state, rates = {}, []
        usage0 = {s_: await _index_usage(req, s_) for s_ in seeds}
        warm_t0 = time.time()
        while time.time() < warm_t0 + warmup:
            r = await throughput_probe(probe_state)
            if r is not None:
                rates.append(r)
            await asyncio.sleep(min(60, max(0.0, warm_t0 + warmup - time.time())))
        target = float(params.get("target-docs-per-s", 0))
        if target and rates:
            tail = sorted(rates[-5:])
            ev("warm_gate", median_docs_per_s=round(tail[len(tail) // 2]), target=target, ok=tail[len(tail) // 2] >= 0.98 * target)
        usage1 = {s_: await _index_usage(req, s_) for s_ in seeds}
        dt = max(1.0, time.time() - warm_t0)
        grow = [
            (v["bytes"] - usage0[i]["shards"][sid]["bytes"]) / dt
            for i, u in usage1.items()
            for sid, v in u["shards"].items()
            if sid in usage0[i]["shards"]
        ]
        g = max(0.0, sum(grow) / len(grow)) if grow else 0.0
        s_max = max([v["bytes"] for u in usage1.values() for v in u["shards"].values()] or [0])
        horizon = float(params.get("horizon-seconds", 18000))
        margin = float(params.get("threshold-margin", 1.3))
        calc_gib = math.ceil(margin * (s_max + g * horizon) / GIB)
        thr_gib = int(_control().get("threshold_gib") or params.get("threshold-gib") or calc_gib)
        cap_gib = int(params.get("max-threshold-gib", 24))
        ev("threshold_calc", seed_max_shard_gib=round(s_max / GIB, 2), growth_bytes_per_s_per_shard=round(g), horizon_s=horizon, margin=margin, calculated_gib=calc_gib, used_gib=thr_gib, cap_gib=cap_gib)
        if thr_gib > cap_gib:
            invalid("threshold_above_cap", threshold_gib=thr_gib, cap_gib=cap_gib)
        # the reshard monitor must be able to read what we read: probe the resharding metadata route before anything depends on it
        detect = _control().get("detect") or params.get("detect-mode", "state")
        rs0 = await _resharding_state(req, split_idx, "state")
        ev("reshard_probe", mode_requested=detect, state_route_readable=rs0 is not None, resharding_now=rs0)
        if rs0 is None and detect == "state":
            attention("resharding_metadata_unreadable_stop_and_ask", hint="set detect to shards-stable in ~/m2s1-control.json only after asking: it is not a safe proxy")
            for _ in range(240):  # up to 2 h: wait for the control file, do not guess
                await asyncio.sleep(30)
                if _control().get("detect"):
                    detect = _control()["detect"]
                    break
            else:
                invalid("resharding_metadata_unreadable")
        await apply_arm_settings()
        await req("PUT", "/_cluster/settings", body={"persistent": {"indices.auto_reshard.shard_size_threshold": "%dgb" % thr_gib}}, retry=True)
        state = await cluster_settings()
        ev("threshold_set", gib=thr_gib, state=state)
        if str(state.get("indices.auto_reshard.shard_size_threshold", "")).lower() not in ("%dgb" % thr_gib, "%dgb" % thr_gib):
            invalid("threshold_not_in_effect", state=state)
        thr = thr_gib * GIB
        base_cluster = {k: v for k, v in state.items() if k.startswith("indices.auto_reshard")}

        async def seed_counts():
            r = await req("GET", "/%s/_settings" % ",".join(seeds), params_={"filter_path": "*.settings.index.number_of_shards"}, retry=True)
            return {k: int(v["settings"]["index"]["number_of_shards"]) for k, v in r.items()}

        seeds_at_start = await seed_counts()
        ev("seed_shard_counts", counts=seeds_at_start, when="start")

        feeder = _Feeder(req, ev, split_idx, corpus, batch=int(params.get("feed-batch", 5000)), concurrency=int(params.get("feed-concurrency", 8)))
        feeder.start()

        async def sizes_event(label, u=None):
            u = u or await _index_usage(req, split_idx)
            ev(
                "split_sizes",
                label=label,
                n_shards=u["n_shards"],
                total_gib=round(u["total"] / GIB, 3),
                avg_gib=round(u["avg"] / GIB, 3),
                deletes_pct=u["deletes_pct"],
                per_shard={str(k): [v["bytes"], v["docs"], v["deleted"]] for k, v in sorted(u["shards"].items())},
            )
            return u

        async def feed_until(pred, label, limit_s):
            t_stop = time.time() + limit_s
            while time.time() < t_stop and time.time() < t_end:
                u = await _index_usage(req, split_idx)
                if pred(u):
                    await feeder.pause(label)
                    return await sizes_event(label + "_reached")
                if _control().get("feed_pause"):
                    await feeder.pause("control")
                else:
                    await feeder.resume(label)
                await asyncio.sleep(20)
            await feeder.pause(label + "_timeout")
            attention("feed_timeout", label=label)
            return await _index_usage(req, split_idx)

        async def put_snapshot(name):
            await req("POST", "/%s/_flush" % split_idx, retry=True)
            await asyncio.sleep(settle_s)  # the tracker evaluates every 30 s: at least a few fresh per-shard lines before the snapshot starts
            u = await sizes_event("before_" + name)
            ev(
                "snapshot_context",
                snapshot=name,
                store_total_bytes=u["total"],
                n_shards=u["n_shards"],
                per_shard_bytes={str(k): v["bytes"] for k, v in sorted(u["shards"].items())},
                resharding=await _resharding_state(req, split_idx, detect),
            )
            await write_shard_map(name)
            await req("PUT", "/_snapshot/%s/%s" % (repo, name), body={"indices": split_idx, "include_global_state": False})
            s0, _ = await wait_snapshot(name)
            names.append(name)
            if s0.get("state") != "SUCCESS":
                attention("snapshot_not_success", snapshot=name, state=s0.get("state"), failures=len(s0.get("failures") or []))
            return s0

        # snapshot 0: the source shard's repository history, taken below the threshold (no split can be running)
        u = await feed_until(lambda u: u["total"] >= float(params.get("pre-split-fraction", 0.8)) * thr, "pre_split", float(params.get("feed-limit-seconds", 7200)))
        await put_snapshot("split-s0")

        async def one_split(k, shards_before):
            """feed until the average per shard is above T, stop writing, wait for the split to start and to finish, settle, snapshot."""
            over = float(params.get("overshoot-fraction", 1.005))
            t_cross = None
            expunged = False
            t_phase = time.time()
            while time.time() < t_end:
                u = await _index_usage(req, split_idx)
                rs = await _resharding_state(req, split_idx, detect)
                if rs or u["n_shards"] > shards_before:
                    break
                ctl = _control()
                if u["avg"] >= over * thr:
                    await feeder.pause("avg_over_threshold")
                    if t_cross is None:
                        t_cross = time.time()
                        await sizes_event("crossed_%d" % k, u)
                    waited = time.time() - t_cross
                    if waited > trigger_timeout and every("not_triggered_%d" % k, 600):
                        cs = await cluster_settings()
                        attention("split_not_triggered", k=k, waited_s=round(waited), avg_gib=round(u["avg"] / GIB, 2), deletes_pct=u["deletes_pct"], settings=cs)
                elif ctl.get("feed_pause"):
                    await feeder.pause("control")
                else:
                    t_cross = None
                    await feeder.resume("below_threshold_%d" % k)
                # deletes of the previous split not merged away: the monitor skips the index while deletes >= 20%. Ask for them to be expunged once, after a wait
                if shards_before > 1 and not expunged and u["deletes_pct"] >= 18 and (time.time() - t_phase > float(params.get("expunge-wait-seconds", 1800)) or ctl.get("expunge")):
                    expunged = True
                    ev("expunge_deletes", deletes_pct=u["deletes_pct"], response=await req("POST", "/%s/_forcemerge" % split_idx, params_={"only_expunge_deletes": "true", "wait_for_completion": "false"}))
                if every("wait_%d" % k, 60):
                    await sizes_event("wait_%d" % k, u)
                await asyncio.sleep(20)
            else:
                invalid("split_not_started", k=k)
            await feeder.pause("split_started")
            t_start = time.time()
            ev("split_started", k=k, n_shards=u["n_shards"], avg_gib=round(u["avg"] / GIB, 3), crossed_for_s=round(t_start - t_cross) if t_cross else None, resharding=rs)
            await sizes_event("split_%d_started" % k)
            done_deadline = t_start + float(params.get("split-done-timeout-seconds", 3600))
            stable_since = None
            while time.time() < done_deadline:
                rs = await _resharding_state(req, split_idx, detect)
                u = await _index_usage(req, split_idx)
                if detect == "state":
                    if rs is False and u["n_shards"] == shards_before * 2:
                        break
                else:  # shards-stable (not a safe proxy; used only when the control file says so): shard count reached and nothing changed for 10 min
                    if u["n_shards"] == shards_before * 2:
                        stable_since = stable_since or time.time()
                        if time.time() - stable_since > 600:
                            break
                if every("splitting_%d" % k, 60):
                    await sizes_event("splitting_%d" % k, u)
                await asyncio.sleep(10)
            else:
                invalid("split_not_finished", k=k)
            ev("split_done", k=k, duration_s=round(time.time() - t_start, 1), n_shards=u["n_shards"], detect=detect)
            sc = await seed_counts()
            if sc != seeds_at_start:
                attention("seed_index_split", before=seeds_at_start, now=sc)
            await sizes_event("split_%d_done" % k, u)
            # sizes every minute during the settle (deletes and merges), then the snapshot
            t_settle = time.time() + settle_s
            while time.time() < t_settle:
                await sizes_event("settle_%d" % k)
                await asyncio.sleep(min(60, max(0.0, t_settle - time.time())))
            return await put_snapshot("split-s%d" % k)

        await one_split(1, 1)
        await one_split(2, 2)
        await feeder.stop()
        await req("PUT", "/_cluster/settings", body={"persistent": {"indices.auto_reshard.shard_size_threshold": params.get("threshold-reset", "1000gb")}}, retry=True)
        ev("threshold_reset", state=await cluster_settings(), was=base_cluster)
        ev("seed_shard_counts", counts=await seed_counts(), when="end", unchanged=(await seed_counts()) == seeds_at_start)
        ev("phase_b_done", snapshots=names, catchup=None, threshold_gib=thr_gib, feeder_docs=feeder.docs, feeder_rejections=feeder.rejections)

    ev("end", mode=mode)
    return {"weight": 1, "unit": "ops", "success": True}


def register(registry):
    async_runner = registry.meta_data.get("async_runner", False)
    if async_runner:
        registry.register_runner("wait-for-ml-lookback", wait_for_ml_lookback_async, async_runner=True)
        registry.register_runner("snapshot-scaling-controller", snapshot_scaling_controller_async, async_runner=True)
        registry.register_runner("snapshot-oracle", snapshot_oracle_async, async_runner=True)
        registry.register_runner("qos-baseline", qos_baseline_async, async_runner=True)
    else:
        registry.register_runner("wait-for-ml-lookback", wait_for_ml_lookback)
