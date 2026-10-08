"use strict";
document.querySelectorAll(".card").forEach(card => {
  const video = card.querySelector("video");
  card.querySelectorAll("button[data-src]").forEach(button => {
    button.addEventListener("click", () => {
      const position = video.currentTime || 0;
      const resume = !video.paused;
      video.pause();
      video.querySelectorAll("source").forEach(source => source.remove());
      video.src = button.dataset.src;
      video.setAttribute("aria-label", card.querySelector("h2").textContent + " " + button.dataset.kind);
      card.querySelectorAll("button[data-src]").forEach(choice => {
        const selected = choice === button;
        choice.classList.toggle("active", selected);
        choice.setAttribute("aria-pressed", String(selected));
      });
      video.addEventListener("loadedmetadata", () => {
        if (position < video.duration) video.currentTime = position;
        if (resume) video.play().catch(() => {});
      }, {once:true});
      video.load();
    });
  });
});
document.addEventListener("play", event => {
  if (event.target.tagName !== "VIDEO") return;
  document.querySelectorAll("video").forEach(video => {
    if (video !== event.target) video.pause();
  });
}, true);
