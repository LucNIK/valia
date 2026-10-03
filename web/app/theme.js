// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

// Loaded in <head>, before the first paint, so the page never flashes the wrong theme.
// The choice is remembered on this device only; without storage it follows the system.
(function () {
  var KEY = "valia-theme";
  var root = document.documentElement;
  /** @returns {string | null} */
  function stored() {
    try { return localStorage.getItem(KEY); } catch (_) { return null; }
  }
  var media = window.matchMedia("(prefers-color-scheme: dark)");
  /** @param {string} theme */
  function apply(theme) {
    root.setAttribute("data-theme", theme);
    var meta = document.querySelectorAll('meta[name="theme-color"]');
    for (var i = 0; i < meta.length; i++) meta[i].setAttribute("content", theme === "dark" ? "#0d0c0f" : "#f7f6f2");
  }
  apply(stored() || (media.matches ? "dark" : "light"));
  media.addEventListener("change", function (e) { if (!stored()) apply(e.matches ? "dark" : "light"); });
  document.addEventListener("DOMContentLoaded", function () {
    var button = document.getElementById("theme");
    if (!button) return;
    button.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      apply(next);
      try { localStorage.setItem(KEY, next); } catch (_) { /* private mode: not remembered */ }
    });
  });
})();
