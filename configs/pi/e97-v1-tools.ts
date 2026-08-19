import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { readFile, readdir, realpath } from "node:fs/promises";
import { isAbsolute, relative, resolve, sep } from "node:path";

const MAX_READ_BYTES = 8192;
const MAX_RESULTS = 256;

function inside(root: string, candidate: string): boolean {
  const rel = relative(root, candidate);
  return rel === "" || (!rel.startsWith(`..${sep}`) && rel !== ".." && !isAbsolute(rel));
}

async function confinedExistingPath(root: string, requested: string): Promise<string> {
  const candidate = resolve(root, requested.replace(/^@/, ""));
  if (!inside(root, candidate)) throw new Error("path escapes the task root");
  const canonical = await realpath(candidate);
  if (!inside(root, canonical)) throw new Error("resolved path escapes the task root");
  return canonical;
}

function json(value: unknown): string {
  return JSON.stringify(value);
}

export default function e97V1Tools(pi: ExtensionAPI) {
  pi.registerTool({
    name: "calculator",
    label: "Calculator",
    description: "Evaluate one integer expression containing +, -, or *",
    parameters: Type.Object({ expression: Type.String({ maxLength: 80 }) }),
    async execute(_id, { expression }) {
      const match = expression.match(/^\s*(-?\d+)\s*([+*-])\s*(-?\d+)\s*$/);
      if (!match) throw new Error("expression must contain two integers and one +, -, or * operator");
      const left = BigInt(match[1]);
      const right = BigInt(match[3]);
      const result = match[2] === "+" ? left + right : match[2] === "-" ? left - right : left * right;
      return { content: [{ type: "text", text: json({ result: result.toString() }) }], details: {} };
    },
  });

  pi.registerTool({
    name: "search",
    label: "Search Records",
    description: "Find bounded record files whose name or content contains a literal query",
    parameters: Type.Object({ query: Type.String({ minLength: 1, maxLength: 160 }) }),
    async execute(_id, { query }, _signal, _update, ctx) {
      const root = await realpath(ctx.cwd);
      const records = await confinedExistingPath(root, "records");
      const names = (await readdir(records, { withFileTypes: true }))
        .filter((entry) => entry.isFile())
        .map((entry) => entry.name)
        .sort();
      const needle = query.toLowerCase();
      const matches: string[] = [];
      for (const name of names) {
        if (matches.length >= MAX_RESULTS) break;
        const path = resolve(records, name);
        const content = (await readFile(path)).subarray(0, MAX_READ_BYTES).toString("utf8");
        if (name.toLowerCase().includes(needle) || content.toLowerCase().includes(needle)) {
          matches.push(relative(root, path));
        }
      }
      return { content: [{ type: "text", text: json({ matches }) }], details: {} };
    },
  });

  pi.registerTool({
    name: "read",
    label: "Read Bounded File",
    description: `Read at most ${MAX_READ_BYTES} bytes from one task-root file`,
    parameters: Type.Object({ path: Type.String({ minLength: 1, maxLength: 512 }) }),
    async execute(_id, { path }, _signal, _update, ctx) {
      const root = await realpath(ctx.cwd);
      const canonical = await confinedExistingPath(root, path);
      const data = await readFile(canonical);
      if (data.length > MAX_READ_BYTES) throw new Error(`file exceeds ${MAX_READ_BYTES} byte limit`);
      return { content: [{ type: "text", text: json({ content: data.toString("utf8") }) }], details: {} };
    },
  });

  pi.registerTool({
    name: "list",
    label: "List Bounded Directory",
    description: "List at most 256 direct files in a task-root directory, optionally filtered by suffix",
    parameters: Type.Object({
      path: Type.String({ minLength: 1, maxLength: 512 }),
      suffix: Type.Optional(Type.String({ maxLength: 40 })),
    }),
    async execute(_id, { path, suffix }, _signal, _update, ctx) {
      const root = await realpath(ctx.cwd);
      const directory = await confinedExistingPath(root, path);
      const files = (await readdir(directory, { withFileTypes: true }))
        .filter((entry) => entry.isFile() && (!suffix || entry.name.endsWith(suffix)))
        .map((entry) => relative(root, resolve(directory, entry.name)))
        .sort();
      if (files.length > MAX_RESULTS) throw new Error(`directory exceeds ${MAX_RESULTS} result limit`);
      return { content: [{ type: "text", text: json({ files }) }], details: {} };
    },
  });
}
