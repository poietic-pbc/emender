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

interface CliResult {
  argv: string[];
  cwd: string;
  exit_code: number;
  stdout: string;
  stderr: string;
  stdout_truncated: boolean;
  stderr_truncated: boolean;
  timed_out: boolean;
  duration_ms: number;
}

export default function e97CliTools(pi: ExtensionAPI) {
  const observations: CliResult[] = [];

  pi.registerTool({
    name: "cli",
    label: "Sandboxed CLI",
    description: "Run one argv vector in the isolated current-directory sandbox. Use repo --help to discover repository commands.",
    parameters: Type.Object({
      argv: Type.Array(Type.String({ minLength: 1, maxLength: 4096 }), { minItems: 1, maxItems: 64 }),
      timeout_seconds: Type.Optional(Type.Integer({ minimum: 1, maximum: 120 })),
    }),
    async execute(_id, { argv, timeout_seconds }, signal, _update, ctx) {
      if (!image || !imageSha256 || !python) throw new Error("CLI sandbox authority is not configured");
      if (signal.aborted) throw new Error("CLI execution aborted");
      const args = [
        runner, "--image", image, "--image-sha256", imageSha256,
        "--cwd", ctx.cwd, "--timeout", String(timeout_seconds ?? 30),
        "--max-output-bytes", "16384", "--", ...argv,
      ];
      const { stdout } = await execFileAsync(python, args, {
        cwd: ctx.cwd,
        env: process.env,
        timeout: ((timeout_seconds ?? 30) + 10) * 1000,
        maxBuffer: 2 * 1024 * 1024,
        signal,
      });
      const parsed = JSON.parse(stdout) as CliResult;
      if (!Array.isArray(parsed.argv) || typeof parsed.exit_code !== "number") {
        throw new Error("CLI runner returned an invalid result");
      }
      observations.push(parsed);
      if (observations.length > 16) observations.shift();
      let visibleStdout = parsed.stdout;
      if (parsed.exit_code === 0 && parsed.argv.at(-1) === "--help") {
        const lines = parsed.stdout.split("\n");
        const usage: string[] = [];
        for (const line of lines) {
          if (line.startsWith("usage:") || (usage.length > 0 && line.startsWith(" "))) usage.push(line.trim());
          else if (usage.length > 0) break;
        }
        visibleStdout = usage.join(" ").replace(/\s+/g, " ").replace(/\{(?:\d+,){8,}\d+\}/g, "INTEGER") + "\n";
      }
      const stable = parsed.exit_code === 0 && !parsed.timed_out
        ? { ok: true, stdout: visibleStdout }
        : { ok: false, stdout: parsed.stdout, exit_code: parsed.exit_code, stderr: parsed.stderr };
      return { content: [{ type: "text", text: JSON.stringify(stable) }], details: parsed };
    },
  });

  pi.registerTool({
    name: "submit_answer",
    label: "Submit CLI-Grounded Answer",
    description: "Submit a value and exact evidence copied from successful CLI stdout, then finish",
    parameters: Type.Object({
      value: Type.String({ minLength: 1, maxLength: 2048 }),
      evidence: Type.String({ minLength: 1, maxLength: 4096 }),
    }),
    async execute(_id, { value, evidence }) {
      const successful = observations.filter((row) => row.exit_code === 0 && !row.timed_out);
      if (!successful.some((row) => row.stdout.includes(value))) {
        throw new Error("answer value is not present in successful CLI output");
      }
      if (!successful.some((row) => row.stdout.includes(evidence))) {
        throw new Error("answer evidence is not present in successful CLI output");
      }
      observations.length = 0;
      return {
        content: [{ type: "text", text: value }],
        details: { value, evidence, grounded: true },
        terminate: true,
      };
    },
  });
}
