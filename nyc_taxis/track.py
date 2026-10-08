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

    async def req(method, path, body=None, params_=None):
        return _body(await es.perform_request(method=method, path=path, body=body, params=params_))

    async def wait_snapshot(name):
        while time.time() < deadline:
            s0 = (await req("GET", "/_snapshot/%s/%s" % (repo, name))).get("snapshots", [{}])[0]
            if s0.get("state") in ("SUCCESS", "PARTIAL", "FAILED"):
                st = await req("GET", "/_snapshot/%s/%s/_status" % (repo, name))
                tot = st.get("snapshots", [{}])[0].get("stats", {})
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
        while time.time() < deadline:
            h = await req("GET", "/_cluster/health/" + indices, params_={"wait_for_status": "green", "timeout": "30s"})
            if h.get("status") == "green":
                return
            await asyncio.sleep(poll)
        raise TimeoutError("green " + indices)

    async def restore(snapshot, indices, rename_to):
        t0 = time.time()
        ev("restore_start", snapshot=snapshot, indices=indices, rename_to=rename_to)
        await req(
            "POST",
            "/_snapshot/%s/%s/_restore" % (repo, snapshot),
            body={
                "indices": indices,
                "include_global_state": False,
                "rename_pattern": "%s(.+)" % prefix,
                "rename_replacement": rename_to + "$1",
            },
        )
        await wait_green(rename_to + "*")
        ev("restore_done", snapshot=snapshot, rename_to=rename_to, duration_s=round(time.time() - t0, 1))

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
        ev(
            "slm_policies",
            policies={
                k: {"schedule": v.get("policy", {}).get("schedule"), "repository": v.get("policy", {}).get("repository")}
                for k, v in (await req("GET", "/_slm/policy")).items()
            },
        )
        # wait until the index tier has the expected node count for stable-seconds (the floor needs a few minutes)
        want, stable = int(params.get("expected-index-nodes", 3)), float(params.get("stable-seconds", 300))
        prev, since = None, time.time()
        while time.time() < deadline:
            cur = await index_nodes()
            if cur != prev:
                ev("index_nodes", nodes=cur)
                prev, since = cur, time.time()
            if len(cur) == want and time.time() - since >= stable:
                break
            await asyncio.sleep(30)
        # restore smoke test, so a refused restore fails the run in minutes rather than hours
        smoke = prefix + "smoke"
        for i in (smoke, smoke + "-r"):
            await req("DELETE", "/" + i, params_={"ignore_unavailable": "true"})
        await req("PUT", "/" + smoke)
        await req("POST", "/%s/_doc" % smoke, body={"ok": 1}, params_={"refresh": "true"})
        await req("PUT", "/_snapshot/%s/%s" % (repo, smoke), body={"indices": smoke, "include_global_state": False})
        await wait_snapshot(smoke)
        await req(
            "POST",
            "/_snapshot/%s/%s/_restore" % (repo, smoke),
            body={"indices": smoke, "include_global_state": False, "rename_pattern": "(.+)", "rename_replacement": "$1-r"},
        )
        await wait_green(smoke + "-r")
        ev("smoke_ok")
        for i in (smoke, smoke + "-r"):
            await req("DELETE", "/" + i)
        await req("DELETE", "/_snapshot/%s/%s" % (repo, smoke))

    elif mode == "seed":
        await req("POST", "/%s/_flush" % ",".join(seeds))
        await req("PUT", "/_snapshot/%s/seed" % repo, body={"indices": ",".join(seeds), "include_global_state": False})
        s0, tot = await wait_snapshot("seed")
        seed_bytes = tot.get("total", {}).get("size_in_bytes", 0)
        # wall time includes start-up and finalization, so this rate is low and the copy count errs long
        rate = seed_bytes / max(1.0, tot.get("time_in_millis", 0) / 1000)
        target = float(params.get("catchup-target-seconds", 1200))
        copies = min(int(params.get("max-copies", 12)), max(1, math.ceil(target * rate / max(1, seed_bytes))))
        ev(
            "copies_planned",
            seed_gib=round(seed_bytes / 2**30, 2),
            rate_mib_s=round(rate / 2**20, 1),
            copies=copies,
            expected_catchup_s=round(copies * seed_bytes / max(1.0, rate)),
        )
        for k in range(1, copies + 1):
            await restore("seed", ",".join(seeds), "%sc%d-" % (prefix, k))

    elif mode == "phase-b":
        total, warmup = float(params.get("duration-seconds", 7200)), float(params.get("warmup-seconds", 600))
        restore_after = float(params.get("restore-after-catchup-seconds", 600))
        t_end = time.time() + total
        policy = params.get("policy") or next(iter(await req("GET", "/_slm/policy")))
        await asyncio.sleep(warmup)
        ev("slm_start", response=await req("POST", "/_slm/start"), policy=policy)
        first = (await req("POST", "/_slm/policy/%s/_execute" % policy)).get("snapshot_name")
        ev("catchup_started", snapshot=first)
        seen, restored, catchup_end = set(), False, None
        while time.time() < t_end:
            cur = await req("GET", "/_snapshot/%s/_all" % repo, params_={"sort": "start_time", "order": "desc", "size": "5"})
            for s in cur.get("snapshots", []):
                n = s.get("snapshot")
                if n in seen or n == "seed" or s.get("state") not in ("SUCCESS", "PARTIAL", "FAILED"):
                    continue
                seen.add(n)
                await wait_snapshot(n)
                if n == first:
                    catchup_end = time.time()
            if catchup_end and not restored and time.time() - catchup_end >= restore_after:
                restored = True
                await restore("seed", seeds[0], "%sr-" % prefix)
            await asyncio.sleep(30)
        ev("phase_b_done", snapshots=sorted(seen), restored=restored)
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
