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

    async def req(method, path, body=None, params_=None, retry=False):
        # reads and deletes survive a slow or briefly unreachable endpoint. Writes are retried only when the caller says the call is
        # idempotent (retry=True): the settings PUT, _flush, _slm/start. Restores and snapshot creation have their own handling.
        attempts = 6 if (method in ("GET", "DELETE") or retry) else 1
        for i in range(attempts):
            try:
                return _body(await es.perform_request(method=method, path=path, body=body, params=params_))
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
    ev("end", mode=mode)
    return {"weight": 1, "unit": "ops", "success": True}


# ---------------------------------------------------------------------------------------------------------------------
# backlog-qa: QA of the snapshot backlog tracker (elastic/elasticsearch-team#5513, PR 4). Not for merge.
# The runner drives the cluster and logs events (a log line and ~/backlog-qa-events.jsonl). The figures themselves are
# read afterwards from APM, using the event timestamps. See ~/snapshot-scaling-research/m5513-pr4-qa-plan.md.
# Control file ~/backlog-qa-control.json: {"go": true} lets the run past gate 1, {"go": false} ends it (setting off).
# ---------------------------------------------------------------------------------------------------------------------
BQ_SETTING = "stateless.snapshot.backlog.enabled"
BQ_EVENTS = os.path.expanduser("~/backlog-qa-events.jsonl")
BQ_CONTROL = os.path.expanduser("~/backlog-qa-control.json")
BQ_SNAPSHOT_BODY = {"indices": "*", "expand_wildcards": "all", "include_global_state": True}


async def backlog_qa_async(es, params):
    import logging

    try:
        return await _backlog_qa(es, params)
    except BaseException as e:
        logging.getLogger(__name__).exception("backlog-qa {'event': 'runner_failed', 'error': %r}", repr(e)[:300])
        raise


async def _backlog_qa(es, params):
    import logging
    import random

    log = logging.getLogger(__name__)
    settle_s = float(params.get("settle-seconds", 150))
    tail_s = float(params.get("tail-seconds", 120))
    doc_kb = int(params.get("doc-kb", 1))
    acc_mb = int(params.get("acc-mb", 50))
    sizes = [int(x) for x in params.get("shard-steps", [350, 1400])]
    idle_s = float(params.get("idle-seconds", 2400))
    # PRs 5 and 6: a throttled repository whose snapshots run long enough for the master metrics to show the hold
    slow_rate = str(params.get("slow-rate", "2mb"))
    hold_mb = int(params.get("hold-mb", 100))
    hard_stop = time.time() + float(params.get("max-seconds", 28800))
    state = {"nodes_at": 0}

    def ev(_event, /, **kw):
        kw.update(event=_event, t=int(time.time() * 1000), ts=time.strftime("%H:%M:%S", time.gmtime()))
        log.info("backlog-qa %s", kw)
        try:
            with open(BQ_EVENTS, "a") as f:
                f.write(json.dumps(kw, default=str) + "\n")
        except Exception:
            pass

    async def req(method, path, body=None, qp=None):
        return _body(await es.perform_request(method=method, path=path, params=qp or {}, body=body))

    def doc():
        return json.dumps({"n": random.randint(0, 1 << 30), "p": os.urandom(doc_kb * 512).hex()})

    async def bulk(index, n_docs):
        lines = []
        for _ in range(n_docs):
            lines.append('{"index":{}}')
            lines.append(doc())
        r = await req("POST", "/%s/_bulk" % index, body="\n".join(lines) + "\n")
        if r.get("errors"):
            ev("bulk_errors", index=index)

    async def write(index, mb):
        per = max(1, int(1024 / doc_kb))  # about 1 MB
        for _ in range(mb):
            await bulk(index, per)
        await req("POST", "/%s/_refresh" % index)

    async def create(index):
        await req("PUT", "/" + index, body={"settings": {"index.number_of_shards": 1}})

    async def heap():
        r = await req(
            "GET",
            "/_nodes/stats/jvm",
            qp={"filter_path": "nodes.*.name,nodes.*.roles,nodes.*.jvm.mem.heap_used_in_bytes,nodes.*.jvm.mem.heap_max_in_bytes"},
        )
        return {
            n.get("name", i): {"roles": n.get("roles"), "used": n["jvm"]["mem"]["heap_used_in_bytes"], "max": n["jvm"]["mem"]["heap_max_in_bytes"]}
            for i, n in r.get("nodes", {}).items()
        }

    async def primaries():
        r = await req("GET", "/_cat/shards", qp={"format": "json", "h": "index,shard,prirep,state,node", "expand_wildcards": "all"})
        started, other = {}, 0
        for s in r:
            if s["prirep"] == "p":
                if s["state"] == "STARTED":
                    started[s["node"]] = started.get(s["node"], 0) + 1
                else:
                    other += 1
        return {"started_primaries_by_node": started, "primaries_not_started": other}

    async def nodes():
        r = await req("GET", "/_cat/nodes", qp={"format": "json", "h": "name,id,node.role,master,heap.max,ip"})
        m = await req("GET", "/_cat/master", qp={"format": "json"})
        return {"nodes": r, "master": m}

    async def transport_actions():
        # per node, the requests each snapshot backlog action received: count and total size (the report size, PR 5)
        r = await req("GET", "/_nodes/stats/transport", qp={"filter_path": "nodes.*.name,nodes.*.transport.actions"})
        out = {}
        for nid, n in r.get("nodes", {}).items():
            acts = n.get("transport", {}).get("actions", {})
            out[n.get("name", nid)] = {
                a: {"requests": v.get("requests", {}).get("count"), "request_bytes": v.get("requests", {}).get("total_size_in_bytes")}
                for a, v in acts.items()
                if "snapshot/publish_backlog" in a or "snapshot/get_shard_generations" in a
            }
        return out

    async def sample(tag):
        try:
            ev("sample", tag=tag, heap=await heap(), **await primaries())
            if time.time() - state["nodes_at"] > 600:
                ev("nodes", **await nodes())
                ev("transport_actions", tag=tag, actions=await transport_actions())
                state["nodes_at"] = time.time()
        except Exception as e:
            ev("sample_error", tag=tag, error=repr(e)[:200])

    async def nap(seconds, tag):
        end_ = time.time() + seconds
        while time.time() < end_:
            await sample(tag)
            await asyncio.sleep(min(60, max(0.0, end_ - time.time())))

    async def set_enabled(value):
        await req("PUT", "/_cluster/settings", body={"persistent": {BQ_SETTING: value}})
        back = await req("GET", "/_cluster/settings", qp={"flat_settings": "true", "filter_path": "persistent"})
        ev("setting", value=value, read_back=back)

    async def synced(pattern, timeout=240):
        end_ = time.time() + timeout
        while time.time() < end_:
            r = await req(
                "GET",
                "/%s/_stats" % pattern,
                qp={"level": "shards", "filter_path": "indices.*.shards.*.seq_no,indices.*.shards.*.routing.primary"},
            )
            bad = 0
            for idx in r.get("indices", {}).values():
                for copies in idx["shards"].values():
                    for c in copies:
                        if c["routing"]["primary"] and c["seq_no"]["global_checkpoint"] != c["seq_no"]["max_seq_no"]:
                            bad += 1
            if bad == 0:
                return True
            await asyncio.sleep(10)
        ev("checkpoints_not_synced", pattern=pattern)
        return False

    async def merges_idle(pattern, timeout=300):
        end_ = time.time() + timeout
        while time.time() < end_:
            r = await req("GET", "/%s/_stats/merge" % pattern, qp={"filter_path": "_all.primaries.merges.current"})
            if r.get("_all", {}).get("primaries", {}).get("merges", {}).get("current", 0) == 0:
                return True
            await asyncio.sleep(10)
        ev("merges_not_idle", pattern=pattern)
        return False

    async def quiet(pattern="qa-*"):
        await req("POST", "/%s/_flush" % pattern)
        ok1 = await synced(pattern)
        ok2 = await merges_idle(pattern)
        await nap(settle_s, "settle")
        return ok1 and ok2

    async def snapshot(repo, name, writer=None):
        t0 = int(time.time() * 1000)
        try:
            await req("PUT", "/_snapshot/%s/%s" % (repo, name), body=BQ_SNAPSHOT_BODY, qp={"wait_for_completion": "false"})
        except Exception as e:
            ev("snapshot_start_failed", repo=repo, name=name, error=repr(e)[:300])
            raise
        ev("snapshot_started", repo=repo, name=name)
        task = asyncio.ensure_future(writer()) if writer else None
        info = {}
        while True:
            r = await req("GET", "/_snapshot/%s/%s" % (repo, name))
            info = r.get("snapshots", [{}])[0]
            if info.get("state") in ("SUCCESS", "PARTIAL", "FAILED"):
                break
            await asyncio.sleep(5)
        if task:
            task.cancel()
        st = await req("GET", "/_snapshot/%s/%s/_status" % (repo, name))
        s0 = st.get("snapshots", [{}])[0]
        shards = []
        for iname, idx in s0.get("indices", {}).items():
            for sid, sh in idx.get("shards", {}).items():
                inc = sh.get("stats", {}).get("incremental", {})
                if inc.get("size_in_bytes", 0) > 0:
                    shards.append([iname, sid, inc.get("size_in_bytes"), inc.get("file_count")])
        ev(
            "snapshot_done",
            repo=repo,
            name=name,
            t_request=t0,
            state=info.get("state"),
            start=info.get("start_time_in_millis"),
            end=info.get("end_time_in_millis"),
            shards=info.get("shards"),
            incremental_bytes=s0.get("stats", {}).get("incremental", {}).get("size_in_bytes"),
            incremental_files=s0.get("stats", {}).get("incremental", {}).get("file_count"),
            total_bytes=s0.get("stats", {}).get("total", {}).get("size_in_bytes"),
            changed_shards=shards[:200],
            changed_shard_count=len(shards),
        )
        return info

    async def cycle(name, repo, write_fn=None, writer=None):
        ev("cycle_start", cycle=name, repo=repo)
        if write_fn:
            await write_fn()
        ok = await quiet()
        ev("sample_point", cycle=name, repo=repo, quiet=ok)
        await snapshot(repo, "%s-%d" % (name, int(time.time())), writer=writer)
        ev("cycle_done", cycle=name, repo=repo)
        await nap(tail_s, "tail")
        if time.time() > hard_stop:
            raise RuntimeError("hard stop reached")

    async def finish():
        try:
            await set_enabled(None)
        except Exception as e:
            ev("finish_setting_failed", error=repr(e)[:200])
        try:
            await req("POST", "/_slm/start")
        except Exception:
            pass
        ev("end")
        return {"weight": 1, "unit": "ops", "success": True}

    # ---- Phase 0: setup, with the setting off ----
    ev("start", params=params)
    ev("version", info=await req("GET", "/", qp={"filter_path": "version.build_hash,version.number"}), **await nodes())
    ev("cluster_settings", settings=await req("GET", "/_cluster/settings", qp={"flat_settings": "true"}))
    try:
        ev("slm_stop", response=await req("POST", "/_slm/stop"))
    except Exception as e:
        ev("slm_stop_failed", error=repr(e)[:300])
    b = (await req("GET", "/_snapshot/backup"))["backup"]
    for rn in ("qa-a", "qa-b", "qa-slow"):
        st = dict(b.get("settings", {}))
        st["base_path"] = (st.get("base_path", "") + "-" + rn).lstrip("-")
        if rn == "qa-slow":
            st["max_snapshot_bytes_per_sec"] = slow_rate
        await req("PUT", "/_snapshot/" + rn, body={"type": b["type"], "settings": st})
        await req(
            "PUT",
            "/_slm/policy/policy-" + rn,
            body={"schedule": "0 0 0 1 1 ? 2099", "name": "<%s-{now/d}>" % rn, "repository": rn, "config": {"indices": ["*"]}},
        )
        ev("repository_and_policy", repo=rn, type=b["type"], keys=sorted(st.keys()))
    ev("slm_policies", policies=await req("GET", "/_slm/policy", qp={"filter_path": "*.policy.repository,*.next_execution_millis"}))

    # Which indices change on their own (Kibana task manager and the like)?
    idx_list = await req("GET", "/_cat/indices", qp={"format": "json", "expand_wildcards": "all", "h": "index,pri,docs.count,store.size"})
    ev("indices_at_start", indices=idx_list)
    s_a = await req("GET", "/_all/_stats/indexing,store", qp={"expand_wildcards": "all", "filter_path": "indices.*.primaries.indexing.index_total,indices.*.primaries.store.size_in_bytes"})
    await asyncio.sleep(150)
    s_b = await req("GET", "/_all/_stats/indexing,store", qp={"expand_wildcards": "all", "filter_path": "indices.*.primaries.indexing.index_total,indices.*.primaries.store.size_in_bytes"})
    changing = [
        i
        for i, v in s_b.get("indices", {}).items()
        if v["primaries"]["indexing"]["index_total"] != s_a.get("indices", {}).get(i, {}).get("primaries", {}).get("indexing", {}).get("index_total")
    ]
    ev("self_changing_indices", indices=changing)

    ev("baseline_heap_start")
    await nap(600, "baseline_off")
    await set_enabled(True)
    await nap(120, "just_enabled")

    # ---- Phase A ----
    async def make_acc():
        for i in range(1, 7):
            await create("qa-acc-%d" % i)
        for i in range(1, 7):
            await write("qa-acc-%d" % i, acc_mb)

    await cycle("c01-first", "qa-a", write_fn=make_acc)
    ev("gate1_ready")
    end_ = time.time() + 3 * 3600
    go = None
    while time.time() < end_ and go is None:
        try:
            with open(BQ_CONTROL) as f:
                go = json.load(f).get("go")
        except Exception:
            await asyncio.sleep(20)
    ev("gate1_released", go=go)
    if go is not True:
        return await finish()

    async def w(idxs, mb=acc_mb):
        for i in idxs:
            await write("qa-acc-%d" % i, mb)

    async def del_upd():
        await req("POST", "/qa-acc-5/_delete_by_query", body={"query": {"range": {"n": {"lt": (1 << 30) // 10}}}}, qp={"refresh": "true", "conflicts": "proceed"})
        await req(
            "POST",
            "/qa-acc-5/_update_by_query",
            body={"script": {"source": "ctx._source.u = 1"}, "query": {"range": {"n": {"gte": (1 << 30) // 10, "lt": (1 << 30) // 5}}}},
            qp={"refresh": "true", "conflicts": "proceed"},
        )

    async def many_segments():
        for _ in range(20):
            await bulk("qa-acc-6", 512)
            await req("POST", "/qa-acc-6/_refresh")

    async def merge_by_policy():
        try:
            await req("PUT", "/qa-acc-6/_settings", body={"index.merge.policy.segments_per_tier": 2})
        except Exception as e:
            ev("segments_per_tier_failed", error=repr(e)[:200])
        await req("POST", "/qa-acc-6/_forcemerge")  # no max_num_segments, so the state id stays
        await merges_idle("qa-acc-6")

    async def overlap_writer():
        while True:
            await bulk("qa-acc-2", 1024)
            await asyncio.sleep(2)

    await cycle("c02-ctl", "qa-a")
    await cycle("c03-write1", "qa-a", write_fn=lambda: w([1]))
    await cycle("c04-ctl", "qa-a")
    await cycle("c05-write3", "qa-a", write_fn=lambda: w([2, 3, 4]))
    await cycle("c06-ctl", "qa-a")
    await cycle("c07-delete-update", "qa-a", write_fn=del_upd)
    await cycle("c08-ctl", "qa-a")
    await cycle("c09-segments", "qa-a", write_fn=many_segments)
    await cycle("c09b-merge", "qa-a", write_fn=merge_by_policy)
    await cycle("c10-b-first", "qa-b")
    await cycle("c11-two-a", "qa-a", write_fn=lambda: w([1]))
    await cycle("c12-two-b", "qa-b")
    # delete the oldest snapshot of qa-a
    listing = await req("GET", "/_snapshot/qa-a/_all", qp={"filter_path": "snapshots.snapshot,snapshots.start_time_in_millis"})
    oldest = sorted(listing.get("snapshots", []), key=lambda s: s["start_time_in_millis"])[0]["snapshot"]
    ev("delete_started", repo="qa-a", name=oldest)
    await req("DELETE", "/_snapshot/qa-a/" + oldest)
    ev("delete_done", repo="qa-a", name=oldest)
    await nap(tail_s, "tail")
    await cycle("c13-after-delete", "qa-a")
    await cycle("c14-overlap", "qa-a", writer=overlap_writer)
    await cycle("c15-after-overlap", "qa-a")

    # ---- PRs 5 and 6: the hold on the master, with snapshots of qa-slow that run for minutes ----
    async def move_one(index):
        cur_ = await nodes()
        idx_nodes = [n for n in cur_["nodes"] if "I" in n.get("node.role", "")]
        if len(idx_nodes) < 2:
            ev("hold_move_skipped", error="one index node")
            return
        try:
            p_ = await req("GET", "/_cat/shards/" + index, qp={"format": "json", "h": "index,shard,prirep,node,state"})
            src_ = [x for x in p_ if x["prirep"] == "p"][0]["node"]
            dst_ = [n["name"] for n in idx_nodes if n["name"] != src_][0]
            ev("hold_move", index=index, frm=src_, to=dst_, response=await req(
                "POST", "/_cluster/reroute", body={"commands": [{"move": {"index": index, "shard": 0, "from_node": src_, "to_node": dst_}}]}))
        except Exception as e:
            ev("hold_move_skipped", error=repr(e)[:300])

    async def hold_cycle(name, write_idxs=(), second=False, move=False):
        ev("cycle_start", cycle=name, repo="qa-slow")
        if write_idxs:
            await w(list(write_idxs), hold_mb)
        ok = await quiet()
        ev("sample_point", cycle=name, repo="qa-slow", quiet=ok)
        tag_ = int(time.time())
        first_ = asyncio.ensure_future(snapshot("qa-slow", "%s-a-%d" % (name, tag_), writer=overlap_writer))
        later = []
        if second:
            await asyncio.sleep(5)
            later.append(asyncio.ensure_future(snapshot("qa-slow", "%s-b-%d" % (name, tag_))))
        if move:
            await asyncio.sleep(90)
            await move_one("qa-acc-3")
        await first_
        for t_ in later:
            await t_
        ev("cycle_done", cycle=name, repo="qa-slow")
        await nap(tail_s, "tail")

    # first snapshot of qa-slow: everything uploads, slowly; a writer runs and one primary moves during it
    await hold_cycle("h01-hold", move=True)
    # two snapshots back to back: the second one's shards wait for a figure at the generation the first one writes
    await hold_cycle("h02-overlap", write_idxs=(2, 4), second=True)
    await cycle("h03-after-hold", "qa-slow")
    cur = await nodes()
    index_nodes = [n for n in cur["nodes"] if "I" in n.get("node.role", "")]
    if len(index_nodes) >= 2:
        try:
            p = await req("GET", "/_cat/shards/qa-acc-3", qp={"format": "json", "h": "index,shard,prirep,node,state"})
            src = [s for s in p if s["prirep"] == "p"][0]["node"]
            dst = [n["name"] for n in index_nodes if n["name"] != src][0]
            ev("reroute", frm=src, to=dst, response=await req("POST", "/_cluster/reroute", body={"commands": [{"move": {"index": "qa-acc-3", "shard": 0, "from_node": src, "to_node": dst}}]}))
            await nap(settle_s, "after_reroute")
            await cycle("c16-after-relocation", "qa-a")
        except Exception as e:
            ev("reroute_skipped", error=repr(e)[:300])
    else:
        ev("reroute_skipped", error="one index node")

    # ---- Phase B ----
    have = 0

    async def make_small(i):
        name = "qa-s-%d" % i
        await create(name)
        for _ in range(5):
            await bulk(name, 200)
        await req("POST", "/%s/_refresh" % name)

    for target in sizes:
        ev("phase_b_step", target=target)
        sem = asyncio.Semaphore(8)

        async def one(i):
            async with sem:
                await make_small(i)

        await asyncio.gather(*[one(i) for i in range(have, target)])
        have = target
        await req("GET", "/_cluster/health/qa-s-*", qp={"wait_for_status": "green", "timeout": "20m", "wait_for_no_relocating_shards": "true"})
        ev("b_created", target=target, **await nodes())
        await cycle("b%d-snap" % target, "qa-a")
        listing = await req("GET", "/_snapshot/qa-a/_all", qp={"filter_path": "snapshots.snapshot,snapshots.start_time_in_millis"})
        snaps = sorted(listing.get("snapshots", []), key=lambda s: s["start_time_in_millis"])
        victim = snaps[0]["snapshot"]
        ev("delete_started", repo="qa-a", name=victim, step=target)
        await req("DELETE", "/_snapshot/qa-a/" + victim)
        ev("delete_done", repo="qa-a", name=victim, step=target)
        await nap(settle_s, "after_delete")
        ev("toggle_off", step=target)
        await set_enabled(False)
        await nap(120, "toggled_off")
        await set_enabled(True)
        ev("toggle_on", step=target)
        await nap(300, "toggled_on")
        if time.time() > hard_stop:
            return await finish()

    ev("idle_start")
    await nap(idle_s, "idle")

    # ---- Off switch ----
    ev("off_start")
    await set_enabled(None)
    await nap(240, "off")
    await set_enabled(True)
    ev("on_again")
    await nap(300, "on_again")
    return await finish()


def register(registry):
    async_runner = registry.meta_data.get("async_runner", False)
    if async_runner:
        registry.register_runner("wait-for-ml-lookback", wait_for_ml_lookback_async, async_runner=True)
        registry.register_runner("snapshot-scaling-controller", snapshot_scaling_controller_async, async_runner=True)
        registry.register_runner("snapshot-oracle", snapshot_oracle_async, async_runner=True)
        registry.register_runner("qos-baseline", qos_baseline_async, async_runner=True)
        registry.register_runner("backlog-qa", backlog_qa_async, async_runner=True)
    else:
        registry.register_runner("wait-for-ml-lookback", wait_for_ml_lookback)
