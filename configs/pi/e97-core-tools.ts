import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const execFileAsync = promisify(execFile);
const runner = resolve(dirname(fileURLToPath(import.meta.url)), "../../scripts/run_e97_sandbox_cli.py");
const image = process.env.EMENDER_CLI_IMAGE;
const imageSha256 = process.env.EMENDER_CLI_IMAGE_SHA256;
const python = process.env.EMENDER_PYTHON;

interface Result {
  argv: string[];
  exit_code: number;
  stdout: string;
  stderr: string;
  timed_out: boolean;
}

async function run(argv: string[], cwd: string, signal: AbortSignal, timeout = 30): Promise<Result> {
  if (!image || !imageSha256 || !python) throw new Error("Pi core-tool sandbox authority is not configured");
  const args = [
    runner, "--image", image, "--image-sha256", imageSha256,
    "--cwd", cwd, "--timeout", String(timeout), "--max-output-bytes", "16384", "--", ...argv,
  ];
  const { stdout } = await execFileAsync(python, args, {
    cwd, env: process.env, timeout: (timeout + 10) * 1000,
    maxBuffer: 2 * 1024 * 1024, signal,
  });
  const result = JSON.parse(stdout) as Result;
  if (!Array.isArray(result.argv) || typeof result.exit_code !== "number") {
    throw new Error("sandbox runner returned an invalid result");
  }
  return result;
}

function visible(result: Result): string {
  if (result.exit_code === 0 && !result.timed_out) return result.stdout;
  const output = [result.stdout.trimEnd(), result.stderr.trimEnd(), `Command exited with code ${result.exit_code}`]
    .filter(Boolean).join("\n");
  return output;
}

export default function e97CoreTools(pi: ExtensionAPI) {
  pi.registerTool({
    name: "read",
    label: "Read",
    description: "Read a bounded line-numbered slice of a text file in the current directory",
    parameters: Type.Object({
      path: Type.String({ minLength: 1, maxLength: 4096 }),
      offset: Type.Optional(Type.Integer({ minimum: 1 })),
      limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 2000 })),
    }),
    async execute(_id, { path, offset, limit }, signal, _update, ctx) {
      const code = [
        "import pathlib,sys",
        "p=pathlib.Path(sys.argv[1])",
        "o=int(sys.argv[2]); n=int(sys.argv[3])",
        "assert not p.is_absolute() and '..' not in p.parts",
        "lines=p.read_text().splitlines()",
        "print('\\n'.join(f'{i}: {v}' for i,v in enumerate(lines[o-1:o-1+n],o)))",
      ].join(";");
      const result = await run(["python", "-c", code, path, String(offset ?? 1), String(limit ?? 200)], ctx.cwd, signal);
      return { content: [{ type: "text", text: visible(result).trimEnd() }], details: result };
    },
  });

  pi.registerTool({
    name: "bash",
    label: "Bash",
    description: "Run a command inside the hash-pinned cwd-only Apptainer sandbox",
    parameters: Type.Object({ command: Type.String({ minLength: 1, maxLength: 8192 }) }),
    async execute(_id, { command }, signal, _update, ctx) {
      const result = await run(["bash", "-lc", command], ctx.cwd, signal, 120);
      return { content: [{ type: "text", text: visible(result) }], details: result };
    },
  });

  pi.registerTool({
    name: "edit",
    label: "Edit",
    description: "Replace one unique exact text block in a file in the current directory",
    parameters: Type.Object({
      path: Type.String({ minLength: 1, maxLength: 4096 }),
      oldText: Type.String({ minLength: 1, maxLength: 16384 }),
      newText: Type.String({ maxLength: 16384 }),
    }),
    async execute(_id, { path, oldText, newText }, signal, _update, ctx) {
      const code = [
        "import pathlib,sys",
        "p=pathlib.Path(sys.argv[1])",
        "assert not p.is_absolute() and '..' not in p.parts",
        "old=sys.argv[2]; new=sys.argv[3]; text=p.read_text()",
        "assert text.count(old)==1, f'expected one exact block, found {text.count(old)}'",
        "p.write_text(text.replace(old,new))",
        "print(f'Successfully replaced 1 block(s) in {p}.')",
      ].join(";");
      const result = await run(["python", "-c", code, path, oldText, newText], ctx.cwd, signal);
      return { content: [{ type: "text", text: visible(result).trimEnd() }], details: result };
    },
  });

  pi.registerTool({
    name: "write",
    label: "Write",
    description: "Create or overwrite a text file in the current directory",
    parameters: Type.Object({
      path: Type.String({ minLength: 1, maxLength: 4096 }),
      content: Type.String({ maxLength: 16384 }),
    }),
    async execute(_id, { path, content }, signal, _update, ctx) {
      const code = [
        "import pathlib,sys",
        "p=pathlib.Path(sys.argv[1])",
        "assert not p.is_absolute() and '..' not in p.parts",
        "text=sys.argv[2]; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(text)",
        "print(f'Successfully wrote {len(text.encode())} bytes to {p}')",
      ].join(";");
      const result = await run(["python", "-c", code, path, content], ctx.cwd, signal);
      return { content: [{ type: "text", text: visible(result).trimEnd() }], details: result };
    },
  });
}
