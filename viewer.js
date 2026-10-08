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
      video.dataset.view = button.dataset.view;
      video.setAttribute("aria-label", card.querySelector(".video-title").textContent + " " + button.dataset.kind);
      card.querySelector(".video-caption").textContent = button.dataset.view === "analysis"
        ? "추정 내강 경계 및 직경 D (mm)"
        : "좌: 원본 초음파 영상 · 우: 추정 내강 경계 및 직경 D (mm)";
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
