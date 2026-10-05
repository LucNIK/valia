// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

import { readFile } from "node:fs/promises";
import { join } from "node:path";

/**
 * Loader over a local folder, for the tests.
 * @param {string} root
 * @returns {import("../app/assets.js").Loader}
 */
export function fsLoader(root) {
  return {
    async json(path) {
      try {
        return JSON.parse(await readFile(join(root, path), "utf8"));
      } catch (err) {
        if (/** @type {NodeJS.ErrnoException} */ (err).code === "ENOENT") return null;
        throw err;
      }
    },
    async text(path) {
      try {
        return await readFile(join(root, path), "utf8");
      } catch (err) {
        if (/** @type {NodeJS.ErrnoException} */ (err).code === "ENOENT") return null;
        throw err;
      }
    },
    async binary(path) {
      const b = await readFile(join(root, path));
      return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
    },
  };
}
