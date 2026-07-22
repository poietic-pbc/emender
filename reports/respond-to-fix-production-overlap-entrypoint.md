# Rank-level failure-domain correction response

The follow-up is accepted as a required correction to the resilient production
path.  It is not satisfied by the completed generation-overlap admission fix:
the existing design and gap matrix describe node peers/managers as membership
units, while the requested behavior requires every GPU trainer to have an
independent fenced identity, eligibility state, contribution record, and
restart incarnation.

For the two-node/eight-GPU-per-node gate, the implementation must emit all 16
identities as `(node, local_gpu, global_rank, process, manager)`: ranks 0--7 on
node 0 managed by the node-0 manager, and ranks 8--15 on node 1 managed by the
node-1 manager.  Trainer/data-plane membership and quorum are rank based;
manager/native-service health is node local; the holder/leader and publication
fence are control-plane concerns; whole-node loss is a distinct failure case.

The current node-oriented status claims in R02, R03, R05, R06, R09, R11,
R13--R16 and NDP02--NDP17 must be re-audited rather than treated as proof of
rank-level survival.  In particular, the implementation task must empirically
determine whether a single trainer exit/device fault currently causes Slurm or
`srun` to kill the step, poisons a node-local operation, leaves the manager
waiting for eight contributions/applied markers, or can be converted into
explicit non-participation.  No production readiness claim follows from this
response.

The registered successor must implement and test:

- explicit rank leases/incarnations and revocation, with no all-rank wait;
- quorum over eligible rank contributions and exact accepted token/sample
  weights, never nominal world size;
- rejection of post-revocation stale publication;
- later-safe-epoch rejoin under a new incarnation after current-model sync;
- supervisor/Slurm containment of one trainer failure when node retirement is
  not required, while separately testing device-reset/node-corruption and
  whole-node loss;
- rank-level membership, contribution, quorum, rejection, eligibility,
  incarnation, and decision evidence in the accepted-generation certificate
  and durable topology report.

This response conforms to *Resilient DiLoCo Compute Pool*, version 1.  The
successor validation cites the conformance checklist and applicable requirement
IDs R01--R16 and NDP01--NDP17; those requirements are strengthened at the
trainer-rank boundary without weakening the existing manager, native transport,
lease, generation, weighting, fencing, atomicity, or recovery rules.
