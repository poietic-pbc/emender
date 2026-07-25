import socket
import threading
import time

import pytest
import torch

from ndm.resilient_e97_reducer import TensorLayout
from ndm.resilient_pool_runtime import (
    DistributedOwnerServer,
    OwnerEndpoint,
    PoolControlClient,
    PoolControlConfig,
    PoolControlServer,
    PoolStageSLO,
    _bounded_owner_io_timeout,
    _owner_rpc,
    fetch_owned_shards,
    submit_owned_shards,
)


def _port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _state(value):
    return {"flat": torch.arange(value, value + 16,
                                  dtype=torch.float32)}


def test_owner_rpc_io_timeout_scales_with_bounded_frame_size():
    assert _bounded_owner_io_timeout(0, 0, remaining_s=180) == 1.0
    assert _bounded_owner_io_timeout(64 << 20, 0, remaining_s=180) == 8.0
    assert _bounded_owner_io_timeout(0, 64 << 20, remaining_s=180) == 8.0
    assert _bounded_owner_io_timeout(64 << 20, 0, remaining_s=3) == 3.0
    assert _bounded_owner_io_timeout(1 << 40, 0, remaining_s=180) == 15.0


def test_owner_rpc_large_frame_budget_applies_to_established_stream():
    server = DistributedOwnerServer(("127.0.0.1", _port()), "n0",
                                    max_owner_bytes=64 << 20)

    def delayed_dispatch(_request, _payload):
        time.sleep(1.1)
        return {"status": "live", "worker_id": "n0"}, b""

    server.dispatch = delayed_dispatch
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        endpoint = OwnerEndpoint("n0", "i0", "127.0.0.1",
                                 server.server_address[1])
        header, payload = _owner_rpc(
            endpoint, {"op": "ping"}, b"", deadline=time.monotonic() + 3,
            max_payload_bytes=64 << 20)
        assert header["status"] == "live"
        assert payload == b""
    finally:
        server.shutdown()
        server.server_close()


def test_stage_slos_are_derived_from_measured_k40_baseline():
    slo = PoolStageSLO.production()
    assert slo.generation_expected_s == 215
    assert slo.generation_hard_s == 720
    assert slo.first_heartbeat_s == 180
    assert slo.transport_s + slo.freeze_s + slo.apply_s <= 180
    assert slo.generation_hard_s < 4 * slo.generation_expected_s


def test_ready_token_floor_distributed_owner_loss_and_late_join(tmp_path):
    slo = PoolStageSLO(1, 3, 1, 1, 1, 1, 1, 1)
    control = PoolControlServer(
        ("127.0.0.1", _port()),
        PoolControlConfig("run", 7, q_min=2, t_min=10, ready_fraction=.5,
                          base_digest="base", policy_digest="policy",
                          layout_digest="layout", code_digest="code", slo=slo),
        evidence_root=tmp_path / "evidence")
    control_thread = threading.Thread(target=control.serve_forever, daemon=True)
    control_thread.start()
    clients = [PoolControlClient(control.server_address, timeout_s=1).bind("run", 7)
               for _ in range(4)]
    owner_servers = []
    try:
        endpoints = []
        for index in range(3):
            server = DistributedOwnerServer(("127.0.0.1", _port()),
                                            worker_id=f"n{index}", max_owner_bytes=4096)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            owner_servers.append(server)
            endpoint = OwnerEndpoint(f"n{index}", f"i{index}",
                                     "127.0.0.1", server.server_address[1])
            endpoints.append(endpoint)
            clients[index].ready(endpoint, generation=4)

        snapshot = clients[0].open_generation(4, attempt=0,
            deadline=time.monotonic() + 1)
        assert {peer["worker_id"] for peer in snapshot["peers"]} == {"n0", "n1", "n2"}

        # A late peer becomes READY but is not retroactively added to generation 4.
        late_server = DistributedOwnerServer(("127.0.0.1", _port()),
                                             worker_id="n3", max_owner_bytes=4096)
        threading.Thread(target=late_server.serve_forever, daemon=True).start()
        owner_servers.append(late_server)
        clients[3].ready(OwnerEndpoint("n3", "i3", "127.0.0.1",
                                      late_server.server_address[1]), generation=4)
        assert {peer["worker_id"] for peer in clients[3].open_generation(
            4, attempt=0, deadline=time.monotonic() + 1)["peers"]} == {"n0", "n1", "n2"}

        closes = {}
        def contribute(index, tokens):
            closes[index] = clients[index].contribute_and_freeze(
                generation=4, attempt=0, worker_id=f"n{index}", incarnation=f"i{index}",
                contribution_seq=9, accepted_tokens=tokens,
                payload_digest=f"payload-{index}", deadline=time.monotonic() + 2)
        threads = [threading.Thread(target=contribute, args=(0, 3)),
                   threading.Thread(target=contribute, args=(1, 7))]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        close = closes[0]
        assert close["status"] == "commit_ready" and close["accepted_tokens"] == 10
        assert {item["worker_id"] for item in close["frozen_identities"]} == {"n0", "n1"}
        assert close["accepted_payloads"] == {
            "n0": "payload-0", "n1": "payload-1"}
        assert close["exact_tokens_by_worker"] == {"n0": 3, "n1": 7}

        # Native endpoints may begin receiving as soon as the provider is
        # bound, but frames are rejected until the frozen peer route is
        # installed.  No reporter may leave this fenced metadata barrier
        # until every accepted worker has installed every remote route.
        route_ready = {}
        route_release_order = []
        first_reported = threading.Event()

        def await_routes(index):
            if index == 0:
                first_reported.set()
            route_ready[index] = clients[index].await_peer_route_ready(
                generation=4, attempt=0, worker_id=f"n{index}",
                incarnation=f"i{index}", peer_worker_id=f"n{1 - index}",
                peer_incarnation=f"i{1 - index}",
                deadline=time.monotonic() + 2)
            route_release_order.append(index)

        first = threading.Thread(target=await_routes, args=(0,))
        first.start()
        assert first_reported.wait(1)
        time.sleep(.05)
        assert first.is_alive() and route_release_order == []
        second = threading.Thread(target=await_routes, args=(1,))
        second.start()
        first.join(); second.join()
        assert {value["status"] for value in route_ready.values()} == {"ready"}
        assert route_ready[0]["workers"] == ["n0", "n1"]

        # Sharded owners report distinct partial-result roots before direct
        # redistribution. The controller releases the immutable root map only
        # after every frozen owner has reported exactly once.
        owner_roots = {}
        def announce_owner_root(index):
            owner_roots[index] = clients[index].announce_owner_result(
                generation=4, attempt=0, worker_id=f"n{index}",
                incarnation=f"i{index}", result_root=(f"{index + 1:02x}" * 32),
                layout_digest=(f"{index + 3:02x}" * 32),
                global_weight=10, result_bytes=64 + index * 4,
                deadline=time.monotonic() + 2)
        owner_threads = [threading.Thread(target=announce_owner_root, args=(0,)),
                         threading.Thread(target=announce_owner_root, args=(1,))]
        for thread in owner_threads: thread.start()
        for thread in owner_threads: thread.join()
        assert {value["status"] for value in owner_roots.values()} == {"ready"}
        assert owner_roots[0]["roots"] == {"n0": "01" * 32, "n1": "02" * 32}
        assert owner_roots[0]["owners"]["n0"] == {
            "layout_digest": "03" * 32,
            "result_bytes": 64,
            "result_root": "01" * 32,
        }
        with pytest.raises(RuntimeError, match="conflicting owner result replay"):
            clients[0].announce_owner_result(
                generation=4, attempt=0, worker_id="n0", incarnation="i0",
                result_root="03" * 32, layout_digest="03" * 32,
                global_weight=10, result_bytes=64,
                deadline=time.monotonic() + 1)

        layout = TensorLayout.from_state(_state(0), max_chunk_bytes=16)
        contributions = {"n0:i0:9": layout.pack(_state(1)),
                         "n1:i1:9": layout.pack(_state(5))}
        weights = {"n0:i0:9": 3, "n1:i1:9": 7}
        accepted_ids = tuple(sorted(contributions))
        original = tuple(endpoints)
        for server in owner_servers:
            if server.worker_id in {item.worker_id for item in original}:
                server.install(layout, run_id="run", fence=7, generation=4, attempt=0,
                               owners=original, accepted_ids=accepted_ids)

        def send_all(owner_set, attempt=0):
            threads = []
            for contribution_key, contribution_chunks in contributions.items():
                threads.append(threading.Thread(
                    target=submit_owned_shards,
                    kwargs=dict(layout=layout, chunks=contribution_chunks,
                                contribution_id=contribution_key,
                                weight=weights[contribution_key], endpoints=owner_set,
                                run_id="run", fence=7, generation=4, attempt=attempt,
                                deadline=time.monotonic() + 2)))
            for item in threads: item.start()
            for item in threads: item.join()

        send_all(original)
        # Kill a primary after receipt but before redistribution. Sender-retained
        # chunks are then replayed under a deterministic surviving owner map.
        lost_id = layout.owner(0, [item.worker_id for item in original], run_id="run",
                               generation=4, attempt=0)
        lost = next(item for item in owner_servers if item.worker_id == lost_id)
        lost.shutdown(); lost.server_close()
        live = tuple(item for item in original if item.worker_id != lost_id)
        for server in owner_servers:
            if server.worker_id != lost_id:
                server.install(layout, run_id="run", fence=7, generation=4, attempt=0,
                               owners=live, accepted_ids=accepted_ids)
        send_all(live)  # bounded replay from retained sender chunks
        aggregate, metrics = fetch_owned_shards(
            layout=layout, endpoints=live, run_id="run", fence=7, generation=4,
            attempt=0, deadline=time.monotonic() + 2)
        actual = layout.unpack(aggregate)["flat"]
        reference = (_state(1)["flat"].double() * 3 + _state(5)["flat"].double() * 7) / 10
        assert torch.allclose(actual.double(), reference, rtol=0, atol=1e-6)
        assert metrics["redistribution_bytes"] == sum(chunk.nbytes for chunk in aggregate)
        assert len({layout.owner(shard, [item.worker_id for item in live], run_id="run",
                                       generation=4, attempt=0)
                    for shard in range(layout.shard_count)}) > 1

        # Every surviving node independently publishes the identity of the one
        # native result.  The metadata coordinator approves it only after all
        # frozen reporters agree with the exact accepted-token total.
        validated = {}
        def validate_root(index):
            validated[index] = clients[index].validate_result_root(
                generation=4, attempt=0, worker_id=f"n{index}",
                incarnation=f"i{index}", result_root="ab" * 32,
                global_weight=10, result_bytes=64,
                deadline=time.monotonic() + 2)
        root_threads = [threading.Thread(target=validate_root, args=(0,)),
                        threading.Thread(target=validate_root, args=(1,))]
        for thread in root_threads: thread.start()
        for thread in root_threads: thread.join()
        assert {value["status"] for value in validated.values()} == {"validated"}
        assert validated[0]["workers"] == ["n0", "n1"]
        with pytest.raises(RuntimeError, match="conflicting result root replay"):
            clients[0].validate_result_root(
                generation=4, attempt=0, worker_id="n0", incarnation="i0",
                result_root="cd" * 32, global_weight=10, result_bytes=64,
                deadline=time.monotonic() + 1)

        # The missing trainer/peer rejoins only at the next generation, under a
        # new incarnation after catching up to the committed generation.
        clients[2].drain("n2", "i2")
        clients[0].ready(endpoints[0], generation=5)
        clients[1].ready(endpoints[1], generation=5)
        rejoined = OwnerEndpoint("n2", "i2-rejoined", "127.0.0.1",
                                 endpoints[2].port)
        clients[2].ready(rejoined, generation=5)
        next_snapshot = clients[0].open_generation(
            5, attempt=0, deadline=time.monotonic() + 1)
        assert {(peer["worker_id"], peer["incarnation"])
                for peer in next_snapshot["peers"]} == {
                    ("n0", "i0"), ("n1", "i1"), ("n2", "i2-rejoined")}
        with pytest.raises(RuntimeError, match="superseded peer incarnation"):
            clients[2].ready(endpoints[2], generation=5)
    finally:
        control.shutdown(); control.server_close()
        for server in owner_servers:
            try:
                server.shutdown(); server.server_close()
            except Exception:
                pass


def test_async_v21_owner_results_use_exact_tokens_only(tmp_path):
    exact_tokens = 5_245_440
    slo = PoolStageSLO(1, 3, 1, 1, 1, 1, 1, 1)
    control = PoolControlServer(
        ("127.0.0.1", _port()),
        PoolControlConfig("async-v2", 11, q_min=2, t_min=3_934_080,
                          ready_fraction=None, base_digest="base",
                          policy_digest="async-v2-policy",
                          layout_digest="layout", code_digest="code", slo=slo),
        evidence_root=tmp_path / "evidence")
    threading.Thread(target=control.serve_forever, daemon=True).start()
    clients = [
        PoolControlClient(control.server_address, timeout_s=1).bind("async-v2", 11)
        for _ in range(2)
    ]
    try:
        for index, client in enumerate(clients):
            client.ready(
                OwnerEndpoint(f"n{index}", f"i{index}", "127.0.0.1",
                              30000 + index),
                generation=4, run_id="async-v2")
        clients[0].open_generation(4, attempt=2, deadline=time.monotonic() + 1)
        with pytest.raises(RuntimeError, match="forbid aggregation_weight"):
            clients[0]._rpc(
                "contribute", payload=b"payload-forbidden", generation=4,
                attempt=2, worker_id="n0", incarnation="i0",
                contribution_seq=0, accepted_tokens=2_621_440,
                aggregation_weight=18_350_080)
        assert clients[0].contribute(
            4, 2, "n0", "i0", 0, 2_621_440,
            "payload-0")["status"] == "accepted"
        assert clients[1].contribute(
            4, 2, "n1", "i1", 0, 2_624_000,
            "payload-1")["status"] == "accepted"
        close = clients[0]._rpc("close", generation=4, attempt=2)
        assert close["status"] == "commit_ready"
        assert close["accepted_tokens"] == exact_tokens
        assert close["exact_tokens_by_worker"] == {
            "n0": 2_621_440, "n1": 2_624_000}

        owner_key = (4, 2)
        for invalid_weight in (0, exact_tokens - 1, exact_tokens + 1):
            with pytest.raises(RuntimeError, match="owner result metadata is invalid"):
                clients[0].announce_owner_result(
                    generation=4, attempt=2, worker_id="n0", incarnation="i0",
                    result_root="01" * 32, layout_digest="03" * 32,
                    global_weight=invalid_weight, result_bytes=64,
                    deadline=time.monotonic() + 1)
            assert owner_key not in control.owner_results

        waiting = clients[0]._rpc(
            "owner_result", generation=4, attempt=2, worker_id="n0",
            incarnation="i0", result_root="01" * 32,
            layout_digest="03" * 32, global_weight=exact_tokens,
            result_bytes=64)
        assert waiting == {"status": "waiting", "reported": 1, "required": 2}
        prior = dict(control.owner_results[owner_key])
        with pytest.raises(RuntimeError, match="conflicting owner result replay"):
            clients[0].announce_owner_result(
                generation=4, attempt=2, worker_id="n0", incarnation="i0",
                result_root="04" * 32, layout_digest="03" * 32,
                global_weight=exact_tokens, result_bytes=64,
                deadline=time.monotonic() + 1)
        assert control.owner_results[owner_key] == prior

        ready = clients[1].announce_owner_result(
            generation=4, attempt=2, worker_id="n1", incarnation="i1",
            result_root="02" * 32, layout_digest="04" * 32,
            global_weight=exact_tokens, result_bytes=68,
            deadline=time.monotonic() + 1)
        assert ready["status"] == "ready"
        assert ready["global_weight"] == exact_tokens
        assert close["accepted_tokens"] == exact_tokens

        # Result-root consensus accepts the exact frozen token total and no
        # separate lag-derived aggregation envelope.
        control.result_roots.pop(owner_key, None)
        admission = control.admissions[owner_key]

        assert clients[0]._rpc(
            "result_root", generation=4, attempt=2, worker_id="n0",
            incarnation="i0", result_root="05" * 32,
            global_weight=exact_tokens - 1, result_bytes=64) == {
                "status": "waiting", "reported": 1, "required": 2}
        with pytest.raises(
                RuntimeError, match="native result-root token accounting mismatch"):
            clients[1].validate_result_root(
                generation=4, attempt=2, worker_id="n1", incarnation="i1",
                result_root="05" * 32, global_weight=exact_tokens - 1,
                result_bytes=64, deadline=time.monotonic() + 1)
        control.result_roots.pop(owner_key, None)

        validated = {}

        def validate_job_5066162_root(index):
            validated[index] = clients[index].validate_result_root(
                generation=4, attempt=2, worker_id=f"n{index}",
                incarnation=f"i{index}", result_root="50" * 32,
                global_weight=exact_tokens, result_bytes=5_506_770_496,
                deadline=time.monotonic() + 2)

        threads = [
            threading.Thread(target=validate_job_5066162_root, args=(index,))
            for index in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert {item["status"] for item in validated.values()} == {"validated"}
        assert validated[0]["global_weight"] == exact_tokens
        assert admission.close_result.accepted_tokens == exact_tokens
    finally:
        control.shutdown()
        control.server_close()


def test_v21s17_pool_close_waits_for_finite_close_and_includes_preclose_arrivals(
        tmp_path):
    slo = PoolStageSLO(1, 1, 1, 1, 1, 1, 1, 1)
    control = PoolControlServer(
        ("127.0.0.1", _port()),
        PoolControlConfig(
            "scale-v21", 13, q_min=2, t_min=2, ready_fraction=None,
            base_digest="base", policy_digest="policy",
            layout_digest="layout", code_digest="code", slo=slo,
            scale_close_offset_s=.1,
            scale_stable_diversity_floor=2,
            scale_per_ready_worker_token_floor=1,
            scale_closure_digest="a" * 64,
        ),
        evidence_root=tmp_path / "evidence",
    )
    threading.Thread(target=control.serve_forever, daemon=True).start()
    clients = [
        PoolControlClient(control.server_address, timeout_s=1).bind(
            "scale-v21", 13)
        for _ in range(4)
    ]
    try:
        for index, client in enumerate(clients):
            client.ready(
                OwnerEndpoint(
                    f"n{index}", f"i{index}", "127.0.0.1", 31000 + index),
                generation=0,
            )
        opened = clients[0].open_generation(
            0, attempt=1, deadline=time.monotonic() + 1)
        assert opened["scale_closure"]["close_on_q_min"] is False
        assert opened["scale_closure"]["uses_launched_ranks"] is False
        for index in range(2):
            assert clients[index].contribute(
                0, 1, f"n{index}", f"i{index}", 0, 2,
                f"payload-{index}")["status"] == "accepted"
        assert clients[0]._rpc(
            "close", generation=0, attempt=1)["status"] == "open"
        assert clients[2].contribute(
            0, 1, "n2", "i2", 0, 2, "payload-2")["status"] == "accepted"
        time.sleep(.11)
        close = clients[0]._rpc("close", generation=0, attempt=1)
        assert close["status"] == "commit_ready"
        assert {
            item["worker_id"] for item in close["frozen_identities"]
        } == {"n0", "n1", "n2"}
        assert clients[3].contribute(
            0, 1, "n3", "i3", 0, 2,
            "payload-late")["status"] == "rejected_stale_fence"
    finally:
        control.shutdown()
        control.server_close()


def test_four_owner_round_robin_route_readiness_is_reciprocal(tmp_path):
    from scripts.frontier import resilient_e97_role as role

    slo = PoolStageSLO(1, 3, 1, 1, 1, 1, 1, 1)
    control = PoolControlServer(
        ("127.0.0.1", _port()),
        PoolControlConfig("run-4", 3, q_min=4, t_min=4,
                          ready_fraction=None, base_digest="base",
                          policy_digest="policy", layout_digest="layout",
                          code_digest="code", slo=slo),
        evidence_root=tmp_path / "evidence")
    threading.Thread(target=control.serve_forever, daemon=True).start()
    clients = [PoolControlClient(control.server_address, timeout_s=1).bind(
        "run-4", 3) for _ in range(4)]
    endpoints = tuple(
        OwnerEndpoint(f"node-{index}", f"inc-{index}", "127.0.0.1", 30000 + index)
        for index in range(4)
    )
    try:
        for index, endpoint in enumerate(endpoints):
            assert clients[index].ready(
                endpoint, generation=5, run_id="run-4", fence=3)["status"] == "READY"
        for client in clients:
            snapshot = client.open_generation(5, attempt=1,
                                              deadline=time.monotonic() + 1)
            assert len(snapshot["peers"]) == 4

        closes = {}

        def contribute(index):
            closes[index] = clients[index].contribute_and_freeze(
                generation=5, attempt=1, worker_id=f"node-{index}",
                incarnation=f"inc-{index}", contribution_seq=5,
                accepted_tokens=1, payload_digest=f"payload-{index}",
                deadline=time.monotonic() + 2)

        contributors = [threading.Thread(target=contribute, args=(index,))
                        for index in range(4)]
        for thread in contributors:
            thread.start()
        for thread in contributors:
            thread.join()
        assert {close["status"] for close in closes.values()} == {"commit_ready"}

        schedules = {
            index: role._native_remote_endpoints(
                endpoints, local_worker_id=f"node-{index}",
                minimum_contributions=4)
            for index in range(4)
        }
        for round_index in range(3):
            ready = {}

            def report(index):
                peer = schedules[index][round_index]
                ready[index] = clients[index].await_peer_route_ready(
                    generation=5, attempt=1, worker_id=f"node-{index}",
                    incarnation=f"inc-{index}",
                    peer_worker_id=peer.worker_id,
                    peer_incarnation=peer.incarnation,
                    deadline=time.monotonic() + 2)

            reporters = [threading.Thread(target=report, args=(index,))
                         for index in range(4)]
            for thread in reporters:
                thread.start()
            for thread in reporters:
                thread.join()
            assert {result["status"] for result in ready.values()} == {"ready"}

        assert len(control.route_readiness) == 6
    finally:
        control.shutdown()
        control.server_close()


def test_stale_duplicate_and_corrupt_contribution_receipts(tmp_path):
    slo = PoolStageSLO(1, 3, 1, 1, 1, 1, 1, 1)
    server = PoolControlServer(("127.0.0.1", _port()), PoolControlConfig(
        "run", 2, q_min=1, t_min=1, ready_fraction=None, base_digest="b",
        policy_digest="p", layout_digest="l", code_digest="c", slo=slo),
        evidence_root=tmp_path)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    client = PoolControlClient(server.server_address, timeout_s=1).bind("run", 2)
    owner = DistributedOwnerServer(("127.0.0.1", _port()), "n0", max_owner_bytes=64)
    threading.Thread(target=owner.serve_forever, daemon=True).start()
    try:
        client.ready(OwnerEndpoint("n0", "inc", "127.0.0.1", owner.server_address[1]), 0)
        client.open_generation(0, 0, deadline=time.monotonic() + 1)
        first = client.contribute(0, 0, "n0", "inc", 0, 1, "x")
        assert first["status"] == "accepted"
        assert client.contribute(0, 0, "n0", "inc", 0, 1, "x") == first
        assert client.contribute(0, 0, "n0", "inc", 0, 2, "changed")["status"] \
            == "rejected_conflicting_duplicate"
        assert client.contribute(0, 0, "n0", "inc", -1, 1, "x")["status"] \
            == "rejected_corrupt"
        assert client.contribute(0, 1, "n0", "inc", 1, 1, "late")["status"] \
            == "rejected_stale_fence"
    finally:
        server.shutdown(); server.server_close(); owner.shutdown(); owner.server_close()
