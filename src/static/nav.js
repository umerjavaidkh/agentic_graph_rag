/*
 * nav.js — the one top-level tab strip shared by every page in the app.
 *
 * The four pages (chat, upload, graph inspector, feedback) grew separately
 * and each carries its own palette, so this bar defines every colour it uses
 * rather than inheriting page variables that mean different things on each
 * page: --accent is purple on chat and sky blue on feedback, and --border
 * does not exist at all on upload.
 *
 * Include with a single line before </body>:
 *
 *     <script src="/static/nav.js"></script>
 *
 * Layout contract: the bar publishes its own height as --app-nav-h on
 * :root. Pages that size themselves against the viewport subtract it:
 *
 *     height: calc(100vh - 40px - var(--app-nav-h, 0px));
 *
 * The 0px fallback matters -- if this script fails to load, those pages keep
 * exactly the geometry they had before the bar existed, rather than
 * collapsing to a broken layout.
 */
(function () {
  "use strict";

  var NAV_HEIGHT = 44;

  // href is matched against the current path to pick the active tab, so each
  // entry's `match` lists every path that page is reachable at: the friendly
  // route and the static file behind it.
  var TABS = [
    { label: "Chat", href: "/chat", match: ["/chat", "/static/chat.html", "/"] },
    { label: "Ingestion", href: "/upload", match: ["/upload", "/static/upload.html"] },
    {
      label: "Graph Inspector",
      href: "/graph-inspector",
      match: ["/graph-inspector", "/static/graph_inspector.html"],
    },
    { label: "Feedback", href: "/feedback", match: ["/feedback", "/static/feedback.html"] },
  ];

  // Not tabs: /docs is FastAPI's own Swagger page and /health returns JSON,
  // so neither can carry this bar. They open in a new tab instead of
  // navigating away from an app the user is working in.
  var LINKS = [
    { label: "API docs", href: "/docs" },
    { label: "Health", href: "/health" },
  ];

  function styles() {
    return [
      ":root { --app-nav-h: " + NAV_HEIGHT + "px; }",
      // Body is display:flex on chat and a grid host elsewhere, so a bar left
      // in normal flow would become a layout item of whatever each page
      // happens to be. Fixed takes it out of flow entirely and the
      // padding-top added in build() reserves its space instead.
      "body { box-sizing: border-box; }",
      ".app-nav {",
      "  position: fixed; top: 0; left: 0; right: 0; z-index: 9000;",
      "  height: var(--app-nav-h); box-sizing: border-box;",
      "  display: flex; align-items: stretch; gap: 0;",
      "  background: #0b1020; border-bottom: 1px solid #232a3d;",
      "  font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;",
      "  overflow-x: auto; scrollbar-width: none;",
      "}",
      ".app-nav::-webkit-scrollbar { display: none; }",
      ".app-nav__brand {",
      "  display: flex; align-items: center; gap: 0.5rem;",
      "  padding: 0 1rem 0 1.1rem; white-space: nowrap;",
      "  color: #f2eefb; font-size: 0.86rem; font-weight: 650;",
      "  letter-spacing: 0.01em;",
      "}",
      ".app-nav__dot {",
      "  width: 8px; height: 8px; border-radius: 50%;",
      "  background: #6c5ce7; box-shadow: 0 0 10px rgba(108,92,231,0.75);",
      "}",
      ".app-nav__tabs { display: flex; align-items: stretch; }",
      ".app-nav a {",
      "  display: flex; align-items: center; padding: 0 0.95rem;",
      "  color: #9aa3bd; text-decoration: none; white-space: nowrap;",
      "  font-size: 0.83rem; border-bottom: 2px solid transparent;",
      "  transition: color 0.12s ease, background 0.12s ease;",
      "}",
      ".app-nav a:hover { color: #e7e3f5; background: rgba(255,255,255,0.045); }",
      ".app-nav a.is-active {",
      "  color: #fff; border-bottom-color: #6c5ce7;",
      "  background: rgba(108,92,231,0.14);",
      "}",
      ".app-nav__spacer { flex: 1 1 auto; min-width: 0.5rem; }",
      ".app-nav__links { display: flex; align-items: stretch; padding-right: 0.4rem; }",
      ".app-nav__links a { font-size: 0.78rem; color: #6f7896; }",
      "@media (max-width: 620px) {",
      "  .app-nav__brand span:not(.app-nav__dot) { display: none; }",
      "  .app-nav a { padding: 0 0.7rem; font-size: 0.79rem; }",
      "}",
    ].join("\n");
  }

  function isActive(tab, path) {
    // Longest match wins so "/" (listed on Chat) never beats a real page.
    return tab.match.some(function (m) {
      return m === "/" ? path === "/" : path === m || path.indexOf(m) === 0;
    });
  }

  function build() {
    var path = window.location.pathname;
    var nav = document.createElement("nav");
    nav.className = "app-nav";
    nav.setAttribute("aria-label", "Sections");

    var brand = document.createElement("div");
    brand.className = "app-nav__brand";
    var dot = document.createElement("span");
    dot.className = "app-nav__dot";
    brand.appendChild(dot);
    var name = document.createElement("span");
    name.textContent = "Agentic Graph RAG";
    brand.appendChild(name);
    nav.appendChild(brand);

    var tabs = document.createElement("div");
    tabs.className = "app-nav__tabs";
    var best = null;
    TABS.forEach(function (tab) {
      if (isActive(tab, path) && (!best || tab.href.length > best.href.length)) {
        best = tab;
      }
    });
    TABS.forEach(function (tab) {
      var a = document.createElement("a");
      a.href = tab.href;
      a.textContent = tab.label;
      if (tab === best) {
        a.className = "is-active";
        a.setAttribute("aria-current", "page");
      }
      tabs.appendChild(a);
    });
    nav.appendChild(tabs);

    var spacer = document.createElement("div");
    spacer.className = "app-nav__spacer";
    nav.appendChild(spacer);

    var links = document.createElement("div");
    links.className = "app-nav__links";
    LINKS.forEach(function (link) {
      var a = document.createElement("a");
      a.href = link.href;
      a.textContent = link.label;
      a.target = "_blank";
      a.rel = "noopener";
      links.appendChild(a);
    });
    nav.appendChild(links);

    var style = document.createElement("style");
    style.textContent = styles();
    document.head.appendChild(style);
    document.body.insertBefore(nav, document.body.firstChild);

    // Reserve the bar's space by ADDING to whatever padding the page already
    // has, read from the computed style rather than assumed -- chat pads its
    // body by 20px and the others by nothing, and overwriting that would
    // crop the chat shell against the window edge.
    var existing = parseFloat(window.getComputedStyle(document.body).paddingTop) || 0;
    document.body.style.paddingTop = existing + NAV_HEIGHT + "px";
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", build);
  } else {
    build();
  }
})();
