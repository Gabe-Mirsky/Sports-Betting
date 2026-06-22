// Live parlay builder: reads checkbox data attributes, updates the summary panel.
(function () {
  "use strict";

  var boxes = Array.prototype.slice.call(document.querySelectorAll(".pick-box"));
  if (!boxes.length) return;

  var legsEl = document.getElementById("parlay-legs");
  var emptyEl = document.getElementById("parlay-empty");
  var countEl = document.getElementById("m-count");
  var probEl = document.getElementById("m-prob");
  var decEl = document.getElementById("m-dec");
  var amEl = document.getElementById("m-am");
  var saveBtn = document.getElementById("parlay-save");

  function americanFromDecimal(dec) {
    if (!isFinite(dec) || dec <= 1) return "n/a";
    if (dec >= 2) return "+" + Math.round((dec - 1) * 100);
    return "-" + Math.round(100 / (dec - 1));
  }

  function render() {
    var selected = boxes.filter(function (b) { return b.checked; });
    countEl.textContent = selected.length;

    if (!selected.length) {
      emptyEl.style.display = "";
      legsEl.innerHTML = "";
      probEl.textContent = "n/a";
      decEl.textContent = "n/a";
      amEl.textContent = "n/a";
      if (saveBtn) saveBtn.disabled = true;
      return;
    }
    emptyEl.style.display = "none";

    var combined = 1;
    legsEl.innerHTML = "";
    selected.forEach(function (b) {
      var prob = parseFloat(b.getAttribute("data-prob"));
      if (isFinite(prob)) combined *= prob;
      var li = document.createElement("li");
      li.className = "parlay-leg";
      li.innerHTML =
        '<div><div class="leg-main"></div><div class="leg-sub"></div></div>' +
        '<span class="leg-prob"></span>' +
        '<button type="button" class="leg-x" title="Remove">&times;</button>';
      li.querySelector(".leg-main").textContent = b.getAttribute("data-label");
      li.querySelector(".leg-sub").textContent = b.getAttribute("data-match");
      li.querySelector(".leg-prob").textContent = isFinite(prob) ? (prob * 100).toFixed(1) + "%" : "—";
      li.querySelector(".leg-x").addEventListener("click", function () {
        b.checked = false;
        render();
      });
      legsEl.appendChild(li);
    });

    probEl.textContent = (combined * 100).toFixed(1) + "%";
    var dec = combined > 0 ? 1 / combined : Infinity;
    decEl.textContent = isFinite(dec) ? dec.toFixed(2) : "n/a";
    amEl.textContent = americanFromDecimal(dec);
    // Need at least two legs to save a parlay.
    if (saveBtn) saveBtn.disabled = selected.length < 2;
  }

  boxes.forEach(function (b) { b.addEventListener("change", render); });
  render();
})();
