// Address of the solver API.
//   - On this computer (python server/api.py --web): "" = the same server that serves the page.
//   - Anywhere else (GitHub Pages, lunar.rtang.dev): the public API.
// A query parameter overrides both: index.html?api=http://127.0.0.1:8000
// turnstileSiteKey: the Cloudflare Turnstile site key (public, not a secret). Empty = no human check.
// On this computer it is Cloudflare's test key, which always passes.
const LOCAL = ["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
window.LUNAR_CONFIG = {
  api: LOCAL ? "" : "https://lunar-api.rtang.dev",
  turnstileSiteKey: LOCAL ? "1x00000000000000000000AA" : "0x4AAAAAAFYLfzkSO4ICdzj5",
};
