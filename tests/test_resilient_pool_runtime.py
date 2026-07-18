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
