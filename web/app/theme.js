// SPDX-License-Identifier: MIT
// Copyright (c) 2026 John Luke NIKABOU (LucNIK)
// @ts-check

// Loaded in <head>, before the first paint, so the page never flashes the wrong theme.
//
// Three modes, cycled by the button and remembered on this device:
//   auto  — dark when the browser asks for dark OR when it is night (20:00–07:00, local time),
//           light otherwise: gold by day, bordeaux by night. Re-evaluated every minute.
//   light — always light.   dark — always dark.
(function () {
  var KEY = "valia-theme";
  var MODES = ["auto", "light", "dark"];
  var LABELS = { auto: "Thème automatique (navigateur et heure)", light: "Thème clair", dark: "Thème sombre" };
  var NIGHT_FROM = 20;
  var NIGHT_TO = 7;
  var root = document.documentElement;
  var media = window.matchMedia("(prefers-color-scheme: dark)");

  /**
   * @param {string} mode
   * @param {boolean} prefersDark
   * @param {number} hour 0-23, local time
   * @returns {"light" | "dark"}
   */
  function resolve(mode, prefersDark, hour) {
    if (mode === "light" || mode === "dark") return mode;
    var night = hour >= NIGHT_FROM || hour < NIGHT_TO;
    return prefersDark || night ? "dark" : "light";
  }

  /** @returns {string} */
  function storedMode() {
    try {
      var v = localStorage.getItem(KEY);
      return v && MODES.indexOf(v) >= 0 ? v : "auto";
    } catch (_) {
      return "auto";
    }
  }

  var mode = storedMode();

  function apply() {
    var theme = resolve(mode, media.matches, new Date().getHours());
    root.setAttribute("data-theme", theme);
    root.setAttribute("data-theme-mode", mode);
    var meta = document.querySelectorAll('meta[name="theme-color"]');
    for (var i = 0; i < meta.length; i++) meta[i].setAttribute("content", theme === "dark" ? "#0d0c0f" : "#f7f6f2");
    var button = document.getElementById("theme");
    if (button) {
      var label = LABELS[/** @type {"auto" | "light" | "dark"} */ (mode)];
      button.setAttribute("aria-label", label + " — changer");
      button.setAttribute("title", label);
    }
  }

  apply();
  media.addEventListener("change", apply);
  setInterval(apply, 60000);
  document.addEventListener("visibilitychange", function () { if (!document.hidden) apply(); });
  document.addEventListener("DOMContentLoaded", function () {
    apply();
    var button = document.getElementById("theme");
    if (!button) return;
    button.addEventListener("click", function () {
      mode = MODES[(MODES.indexOf(mode) + 1) % MODES.length];
      try { localStorage.setItem(KEY, mode); } catch (_) { /* private mode: not remembered */ }
      apply();
    });
  });

  /** @type {any} */ (window).valiaTheme = { resolve: resolve };
})();
