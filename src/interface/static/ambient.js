/* ambient.js — the drifting background layer, and its lean toward the cursor.
 *
 * Injected rather than pasted into five pages, for the same reason nav.js
 * is: one place to change it.
 *
 * Interaction is deliberately weak. The layer leans a few pixels toward the
 * pointer over more than a second, so it reads as the page being aware of
 * you rather than as something following the mouse -- anything faster
 * competes with what you are reading.
 *
 * Costs: one rAF per pointer move at most, writing two CSS variables.
 * No layout is read, so it cannot cause a synchronous reflow.
 */
(function () {
  "use strict";

  if (document.querySelector(".ambient")) return;

  var reduced = window.matchMedia &&
                window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var layer = document.createElement("div");
  layer.className = "ambient";
  layer.setAttribute("aria-hidden", "true");
  layer.innerHTML = "<span></span><span></span><span></span>";
  // First child of body so it sits behind content without needing a
  // stacking context on anything else.
  document.body.insertBefore(layer, document.body.firstChild);

  if (reduced) return;

  // Coarse pointers have no hover to track, and phones gain nothing but
  // battery drain from this.
  if (window.matchMedia && window.matchMedia("(pointer: coarse)").matches) return;

  var pending = false;
  var x = 0;
  var y = 0;

  function apply() {
    pending = false;
    // A few pixels at the extremes -- the whole effect is peripheral.
    layer.style.transform = "translate3d(" + (x * 14).toFixed(2) + "px," +
                            (y * 14).toFixed(2) + "px,0)";
  }

  window.addEventListener("pointermove", function (e) {
    // Normalised to -0.5..0.5 from the viewport centre, using the event's
    // own coordinates so nothing is measured off the DOM.
    x = e.clientX / window.innerWidth - 0.5;
    y = e.clientY / window.innerHeight - 0.5;
    if (!pending) {
      pending = true;
      requestAnimationFrame(apply);
    }
  }, { passive: true });
})();
