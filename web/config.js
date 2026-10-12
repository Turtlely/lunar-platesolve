// Address of the solver API.
//   - On this computer (python server/api.py --web): "" = the same server that serves the page.
//   - Anywhere else (GitHub Pages, lunar.rtang.dev): the public API.
// A query parameter overrides both: index.html?api=http://127.0.0.1:8000
window.LUNAR_CONFIG = {
  api: ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname) ? "" : "https://lunar-api.rtang.dev",
};
