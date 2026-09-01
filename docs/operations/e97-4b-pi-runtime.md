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
