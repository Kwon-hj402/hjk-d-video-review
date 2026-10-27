"use strict";
const parameters=new URLSearchParams(location.search);
const candidates=parameters.get("set")==="candidates";
const search=document.querySelector("#frame-search");
const splitFilter=document.querySelector("#split-filter");
const labelFilter=document.querySelector("#label-filter");
const results=document.querySelector("#gallery-results");
const previous=document.querySelector("#previous-page"),next=document.querySelector("#next-page");
let records=[],page=0;
const perPage=candidates?18:8;
const escapeHtml=value=>String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const image=(src,label)=>`<figure><figcaption>${label}</figcaption><a href="${escapeHtml(src)}"><img src="${escapeHtml(src)}" alt="${label}" loading="lazy" width="640" height="448"></a></figure>`;
function filtered(){const q=search.value.trim().toLowerCase();return records.filter(r=>(!q||[r.id??r.frame_id,r.source_video].join(" ").toLowerCase().includes(q))&&(candidates||!splitFilter.value||r.split===splitFilter.value)&&(candidates||!labelFilter.value||(labelFilter.value==="negative"?r.negative:!r.negative)));}
function render(){const rows=filtered(),pages=Math.max(1,Math.ceil(rows.length/perPage));page=Math.min(page,pages-1);document.querySelector("#gallery-count").textContent=`전체 ${records.length}장 중 ${rows.length}장`;document.querySelector("#page-number").textContent=`${page+1} / ${pages}`;previous.disabled=page===0;next.disabled=page>=pages-1;const visible=rows.slice(page*perPage,(page+1)*perPage);if(!visible.length){results.textContent="해당하는 프레임이 없습니다.";return;}
results.className=candidates?"candidates-grid":"";
results.innerHTML=visible.map(r=>{if(candidates)return `<article class="candidate"><a href="${escapeHtml(r.image)}"><img src="${escapeHtml(r.image)}" alt="초기 후보 ${escapeHtml(r.frame_id)}" loading="lazy" width="640" height="448"></a><h2>${escapeHtml(r.frame_id)}</h2><p>${escapeHtml(r.source_video)}<br>${Number(r.timestamp_seconds).toFixed(3)}초 · 화질 점수 ${Number(r.quality_score).toFixed(3)}<br>${r.used_in_training_snapshot?"승인 215장 스냅샷에 포함":"학습 스냅샷에 미포함"}</p>${r.used_in_training_snapshot?`<a href="gallery.html?frame=${encodeURIComponent(r.frame_id)}">승인 주석 보기</a>`:""}</article>`;
const metric=r.dice===undefined?"훈련 프레임: 보관된 독립 예측 평가 없음":r.both_empty?"음성 프레임: 참조·예측 모두 비어 있음":`Dice ${r.dice.toFixed(4)} · IoU ${r.iou.toFixed(4)}`;
return `<article class="case" id="${escapeHtml(r.id)}"><h2>${escapeHtml(r.id)} · ${{train:"훈련",val:"검증",test:"시험"}[r.split]} · ${r.negative?"내강 없음":"내강 있음"}</h2><p class="case-meta">${escapeHtml(r.source_video)} · 프레임 ${r.frame_index} · ${r.timestamp_seconds.toFixed(3)}초 · ${r.width}×${r.height}<br>${metric}</p><div class="evidence-images ${r.prediction_comparison?"":"three"}">${image(r.raw,"원본 PNG")}${image(r.annotation_overlay,"사용자 승인 주석")}${image(r.mask_display,"승인 마스크 표시")}${r.prediction_comparison?image(r.prediction_comparison,"청록: 승인 · 주황: 예측"):""}</div><nav class="case-links"><a href="${escapeHtml(r.annotation_json)}">다각형 JSON</a><a href="${escapeHtml(r.yolo_label)}">YOLO 좌표 TXT</a><a href="${escapeHtml(r.mask_binary)}">0/1 마스크 PNG</a>${r.prediction_mask?`<a href="${escapeHtml(r.prediction_mask)}">예측 마스크 PNG</a>`:""}<a href="gallery.html?frame=${encodeURIComponent(r.id)}">이 프레임 링크</a></nav></article>`;
}).join("");}
for(const field of [search,splitFilter,labelFilter])field.addEventListener("input",()=>{page=0;render();});
previous.addEventListener("click",()=>{page--;render();document.querySelector(".gallery-filters").scrollIntoView();});
next.addEventListener("click",()=>{page++;render();document.querySelector(".gallery-filters").scrollIntoView();});
if(candidates){document.querySelector("#gallery-title").textContent="초기 후보 400장";document.querySelector("#gallery-description").textContent="화질·시간 간격·다양성으로 선별한 초기 후보이다. 최종 학습 주석 215장과의 교집합은 27장이다.";document.querySelector("#split-filter-wrap").hidden=true;document.querySelector("#label-filter-wrap").hidden=true;document.querySelector("#gallery-note").textContent="화질 점수는 임상적 적합성이나 내강 레이블을 의미하지 않는다. 원본 전체 크기 PNG를 제공한다.";}
if(parameters.has("frame"))search.value=parameters.get("frame");
if(parameters.has("split"))splitFilter.value=parameters.get("split");
fetch(candidates?"research/candidate400/manifest.json":"research/gallery.json").then(r=>{if(!r.ok)throw Error("자료를 불러오지 못했습니다.");return r.json();}).then(data=>{records=candidates?data.frames:data;render();}).catch(error=>{results.textContent=error.message;document.querySelector("#gallery-count").textContent="불러오기 실패";previous.disabled=true;next.disabled=true;});
