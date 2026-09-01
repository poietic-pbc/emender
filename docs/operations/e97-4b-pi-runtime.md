# E97 4B Pi runtime contract

The E97 4B Pi instruction checkpoints are not generic chat checkpoints. They
were trained against a fixed system prompt, transcript serializer, and closed
four-tool contract. Substituting a generic assistant prompt can produce
repetition or otherwise degenerate output.

## Canonical system prompt

The byte-authoritative, newline-terminated prompt is
[`configs/pi/e97-pi-core-system-prompt.txt`](../../configs/pi/e97-pi-core-system-prompt.txt):

```text
You are a coding agent operating in Pi in the current working directory. Use read, bash, edit, and write to inspect, change, and verify repository files. For a tool call respond with exactly Action and one JSON Arguments object. Never invent tool results. After the work is verified, respond with Final and a concise evidence-grounded summary.
```

The Python authority is `E97_PI_CORE_SYSTEM` in
`ndm/e97_agent_protocol.py`. A regression test requires the published text file
and Python constant to remain identical.

This short prompt—not Pi's full generated system prompt—was serialized into
`pi-native-core-v1` and the later live-aligned repair authorities. Those
synthetic tasks almost always named the correct path in the user request; they
did not teach general repository discovery. Supplying the full Pi prompt is
therefore neither required nor a remedy for the known held-out-family path
substitution failure.

## Assistant turn grammar

The model generates exactly one action or one terminal final per turn:

```text
Action: read
Arguments: {"path":"README.md","offset":1,"limit":80}
```

```text
Final: Verified the requested change and its focused test.
```

`ndm.e97_agent_protocol.parse_agent_turn` is the normative parser. The server
accepts only one JSON object after `Arguments:`, rejects tools not declared by
the OpenAI request, and stops canonical one-line finals at their first newline.

## Pi tools

Use the closed `read`, `bash`, `edit`, and `write` implementation in
[`configs/pi/e97-core-tools.ts`](../../configs/pi/e97-core-tools.ts). Do not
combine it with Pi built-in tools, skills, or repository context files when
reproducing the fixed behavioral panels; those change the prompt and available
actions.

A representative invocation is:

```bash
pi --mode interactive \
  --provider emender-local \
  --model e97-4b-pi \
  --no-builtin-tools \
  --no-skills \
  --no-context-files \
  -e configs/pi/e97-core-tools.ts \
  --system-prompt "$(cat configs/pi/e97-pi-core-system-prompt.txt)"
```

The local OpenAI-compatible server must use `--pi-core-canonical-system` or an
equivalent request-time check. See `scripts/serve_e97_agent_openai.py` and
`ndm/e97_agent_server.py`.

The published model configuration advertises a 32,768-token operational window.
This prevents Pi from reducing `max_completion_tokens` to one after reserving
its own output budget, even when Pi constructs a large prompt before the server
replaces the system message. It is an integration budget, not a claim that this
development checkpoint was behaviorally qualified at 32K.

## Recurrent prefix caching

The server can retain an E97 recurrent state for a Pi session. The published
provider compatibility settings request `x-session-id` affinity headers. On the
first request `x-emender-cache` is `miss`; subsequent append-only requests must
report `hit`, and `x-emender-suffix-tokens` must count only the newly ingested
suffix. A persistent `miss` means session affinity is not reaching the server
and makes the client replay the complete transcript unnecessarily.

A static system prompt can also be prefetched once and cloned as the initial
state for new sessions. Dynamic repository context should remain compact (for
example cwd plus a bounded top-level listing) and be ingested once per session.
Caching removes repeated prefill cost; it does not make an unsupported full Pi
prompt behaviorally equivalent to the short training prompt.

## Transcript serialization

`ndm.e97_agent_protocol.serialize_pi_messages` emits role-labelled sections:

```text
System:
...

User:
...

Assistant:
Action: read
Arguments: {...}

Tool:
...

Assistant:
```

Successful tools with no stdout are represented by the literal Pi observation
`(no tool output)`. Tool failures retain their real bounded stderr and exit
status. Reproductions must preserve these observations; simplified or invented
results change model behavior.

## Reproducibility boundary

The published 119/120 checkpoint is a narrow core-tool milestone, not evidence
of broad repository-level coding ability. Its reported score is valid only with
the checkpoint hash, canonical prompt, tool extension, serializer, sandbox
image, and evaluation authority named by its release receipt.
