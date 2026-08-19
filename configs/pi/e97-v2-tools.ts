import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { readFile, readdir, realpath } from "node:fs/promises";
import { isAbsolute, relative, resolve, sep } from "node:path";

const MAX_READ_BYTES = 8192;
let expectedAnswer: string | null = null;

function inside(root: string, candidate: string): boolean {
  const rel = relative(root, candidate);
  return rel === "" || (!rel.startsWith(`..${sep}`) && rel !== ".." && !isAbsolute(rel));
}

async function confinedFile(root: string, requested: string): Promise<string> {
  const candidate = resolve(root, requested.replace(/^@/, ""));
  if (!inside(root, candidate)) throw new Error("path escapes the task root");
  const canonical = await realpath(candidate);
  if (!inside(root, canonical)) throw new Error("resolved path escapes the task root");
  return canonical;
}

function result(value: unknown) {
  return { content: [{ type: "text" as const, text: JSON.stringify(value) }], details: {} };
}

export default function e97V2Tools(pi: ExtensionAPI) {
  pi.registerTool({
    name: "calculator",
    label: "Calculator",
    description: "Evaluate one integer expression containing +, -, or *",
    parameters: Type.Object({ expression: Type.String({ maxLength: 80 }) }),
    async execute(_id, { expression }) {
      const match = expression.match(/^\s*(-?\d+)\s*([+*-])\s*(-?\d+)\s*$/);
      if (!match) throw new Error("expression must contain two integers and one operator");
      const left = BigInt(match[1]);
      const right = BigInt(match[3]);
      expectedAnswer = (match[2] === "+" ? left + right : match[2] === "-" ? left - right : left * right).toString();
      return result({ expression, value: expectedAnswer });
    },
  });

  pi.registerTool({
    name: "lookup",
    label: "Lookup Record Field",
    description: "Read one bounded record and return exactly one owner or budget field",
    parameters: Type.Object({
      path: Type.String({ minLength: 1, maxLength: 512 }),
      field: Type.Union([Type.Literal("owner"), Type.Literal("budget")]),
    }),
    async execute(_id, { path, field }, _signal, _update, ctx) {
      const root = await realpath(ctx.cwd);
      const canonical = await confinedFile(root, path);
      const data = await readFile(canonical);
      if (data.length > MAX_READ_BYTES) throw new Error(`file exceeds ${MAX_READ_BYTES} byte limit`);
      const text = data.toString("utf8");
      const match = field === "owner"
        ? text.match(/\bowner\s+([^.,\n]+)/i)
        : text.match(/\bbudget\s+(\$[\d,]+)/i);
      if (!match) throw new Error(`record has no ${field} field`);
      expectedAnswer = match[1].trim();
      return result({ field, value: expectedAnswer });
    },
  });

  pi.registerTool({
    name: "count",
    label: "Count Files",
    description: "Count direct files in one bounded directory, optionally filtered by suffix",
    parameters: Type.Object({
      path: Type.String({ minLength: 1, maxLength: 512 }),
      suffix: Type.Optional(Type.String({ maxLength: 40 })),
    }),
    async execute(_id, { path, suffix }, _signal, _update, ctx) {
      const root = await realpath(ctx.cwd);
      const directory = await confinedFile(root, path);
      const entries = await readdir(directory, { withFileTypes: true });
      expectedAnswer = entries.filter((entry) => entry.isFile() && (!suffix || entry.name.endsWith(suffix))).length.toString();
      return result({ count: Number(expectedAnswer) });
    },
  });

  pi.registerTool({
    name: "submit_answer",
    label: "Submit Grounded Answer",
    description: "Submit the exact value from the latest successful tool result and finish",
    parameters: Type.Object({ value: Type.String({ minLength: 1, maxLength: 512 }) }),
    async execute(_id, { value }) {
      if (expectedAnswer === null) throw new Error("no successful tool result is available");
      if (value !== expectedAnswer) throw new Error("answer is not supported by the latest tool result");
      const answer = expectedAnswer;
      expectedAnswer = null;
      return {
        content: [{ type: "text" as const, text: answer }],
        details: { value: answer, grounded: true },
        terminate: true,
      };
    },
  });
}
