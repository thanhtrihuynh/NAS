const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

let candidates = [];
let currentRecord = null;
let selectedBeat = null;
let selectedCandidateId = 69;
let currentSplit = 'test';
let classMatches = [];
let classCursor = -1;
let savedConfigs = [];
let currentConfigName = '';
let currentConfigSnapshot = null;
let nasPollTimer = null;

const roleColors = {
  best_f1:'#f6c85f', balanced:'#4f9cff', lowest_macs:'#27d39b', lowest_params:'#9b6ef3'
};

function fmt(n){ return Number(n).toLocaleString('en-US'); }
function f4(n){ return Number(n).toFixed(4); }
function pct(n){ return `${(Number(n)*100).toFixed(2)}%`; }
function pad4(n){ return String(n).padStart(4,'0'); }
function toast(msg){ const t=$('#toast'); t.textContent=msg; t.classList.add('show'); setTimeout(()=>t.classList.remove('show'),3200); }
async function getJSON(url){ const r=await fetch(url); const j=await r.json(); if(!r.ok) throw new Error(j.error||r.statusText); return j; }
async function postJSON(url,data){ const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}); const j=await r.json(); if(!r.ok) throw new Error(j.error||r.statusText); return j; }

function setView(name){
  $$('.view').forEach(v=>v.classList.toggle('active',v.id===`view-${name}`));
  $$('.nav-btn').forEach(b=>b.classList.toggle('active',b.dataset.view===name));
  window.scrollTo({top:0,behavior:'smooth'});
}

function linePath(points, w, h, pad=10){
  if(!points || !points.length) return '';
  const min=Math.min(...points), max=Math.max(...points), span=(max-min)||1;
  return points.map((v,i)=>{
    const x=pad+(w-2*pad)*(i/(points.length-1||1));
    const y=h-pad-(h-2*pad)*((v-min)/span);
    return `${i?'L':'M'}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
}

function gridSvg(w,h){
  let s='';
  for(let i=1;i<5;i++){ const x=(w*i/5).toFixed(1); s+=`<line class="gridline" x1="${x}" y1="0" x2="${x}" y2="${h}"/>`; }
  for(let i=1;i<4;i++){ const y=(h*i/4).toFixed(1); s+=`<line class="gridline" x1="0" y1="${y}" x2="${w}" y2="${y}"/>`; }
  return s;
}

function renderLargeBeat(svg, points){
  if(!svg) return;
  const w=620,h=220;
  svg.innerHTML=`${gridSvg(w,h)}<path class="wave-line" d="${linePath(points||[],w,h,18)}"/>`;
}

function compressPoints(points, maxPoints=16000){
  if(!points || !points.length) return [];
  const step=Math.max(1, Math.ceil(points.length / maxPoints));
  const out=[];
  for(let i=0;i<points.length;i+=step) out.push(points[i]);
  if(out[out.length-1]!==points[points.length-1]) out.push(points[points.length-1]);
  return out;
}

function buildTimelineData(){
  const beats=currentRecord?.beats||[];
  const all=[];
  const segments=[];
  let cursor=0;
  beats.forEach((beat)=>{
    const pts=(beat.samples||beat.points||[]).map(Number);
    const len=pts.length || 320;
    const start=cursor;
    if(pts.length) all.push(...pts); else for(let i=0;i<len;i++) all.push(0);
    cursor += len;
    segments.push({index:Number(beat.index),label:beat.label,start,end:cursor,length:len});
  });
  return {allPoints:all,segments,totalSamples:cursor};
}

function renderBeatStrip(){
  const overlay=$('#beatOverlay');
  const svg=$('#timelineSvg');
  overlay.innerHTML='';
  if(!currentRecord){ svg.innerHTML=''; return; }

  const timeline=buildTimelineData();
  currentRecord._timeline=timeline;
  const beatCount=currentRecord.beats.length;
  const timelineWidth=Math.max(1100, beatCount * 84);
  const timelineHeight=130;

  svg.setAttribute('viewBox',`0 0 ${timelineWidth} ${timelineHeight}`);
  svg.setAttribute('width',timelineWidth);
  svg.setAttribute('height',timelineHeight);
  overlay.style.width=`${timelineWidth}px`;
  overlay.style.height=`${timelineHeight}px`;
  $('#timelineStage').style.width=`${timelineWidth}px`;
  $('#timelineStage').style.height=`${timelineHeight}px`;

  const compressed=compressPoints(timeline.allPoints);
  svg.innerHTML=`${gridSvg(timelineWidth,timelineHeight)}<path class="wave-line timeline-wave" d="${linePath(compressed,timelineWidth,timelineHeight,10)}"/>`;

  timeline.segments.forEach(seg=>{
    const left=(seg.start/timeline.totalSamples)*timelineWidth;
    const width=((seg.end-seg.start)/timeline.totalSamples)*timelineWidth;
    const box=document.createElement('button');
    box.type='button';
    box.className='beat-box';
    box.dataset.index=seg.index;
    box.dataset.label=seg.label;
    box.style.left=`${left}px`;
    box.style.width=`${Math.max(width,24)}px`;
    box.innerHTML=`<span class="beat-box-num">${seg.index+1}</span><span class="beat-box-label">${seg.label||''}</span>`;
    box.title=`Beat ${seg.index+1} · ${seg.label||''}`;
    box.addEventListener('click',()=>selectBeat(seg.index,true));
    overlay.appendChild(box);
  });

  $('#beatCount').textContent=`${currentRecord.beat_count} beats`;
  setTimeout(()=>{
    const sc=$('#beatScroll');
    const range=$('#beatScrollRange');
    range.max=Math.max(0,sc.scrollWidth-sc.clientWidth);
    range.value=sc.scrollLeft;
  },0);

  applyClassVisuals();
}

function updateClassSelector(){
  const counts=currentRecord?.class_counts||{};
  const sel=$('#classSelect');
  [...sel.options].forEach(opt=>{
    const value=opt.value;
    if(value==='ALL') opt.textContent=`All classes (${currentRecord?.beat_count||0})`;
    else opt.textContent=`${value} (${counts[value]||0})`;
  });
  sel.value='ALL';
  classMatches=[];
  classCursor=-1;
  $('#classMatchInfo').textContent='Choose a class to find matching beats in this record.';
}

function applyClassVisuals(){
  const target=$('#classSelect')?.value || 'ALL';
  $$('.beat-box').forEach(box=>{
    const match=target==='ALL' || box.dataset.label===target;
    box.classList.toggle('class-match',target!=='ALL' && match);
    box.classList.toggle('class-dim',target!=='ALL' && !match);
    box.classList.toggle('selected',selectedBeat && Number(box.dataset.index)===Number(selectedBeat.index));
  });
}

function findClass(first=true){
  if(!currentRecord){ toast('Load an ECG record first.'); return; }
  const target=$('#classSelect').value;
  if(target==='ALL'){
    classMatches=[]; classCursor=-1; applyClassVisuals();
    $('#classMatchInfo').textContent=`Showing all ${currentRecord.beat_count} beats.`;
    return;
  }

  classMatches=currentRecord.beats.filter(b=>String(b.label)===target);
  applyClassVisuals();
  if(!classMatches.length){
    $('#classMatchInfo').textContent=`Class ${target}: 0 beats in record ${currentRecord.record_id}.`;
    toast(`No ${target} beat found in record ${currentRecord.record_id}.`);
    return;
  }

  classCursor = first ? 0 : ((classCursor+1)%classMatches.length);
  const beat=classMatches[classCursor];
  $('#classMatchInfo').textContent=`Class ${target}: ${classMatches.length} beats · showing ${classCursor+1}/${classMatches.length}`;
  selectBeat(beat.index,true);
}

function selectBeat(index, autoScroll=false){
  if(!currentRecord) return;
  selectedBeat=currentRecord.beats.find(b=>Number(b.index)===Number(index));
  if(!selectedBeat) return;

  applyClassVisuals();
  $('#selectedBeatTop').textContent=`Selected: Beat ${selectedBeat.index+1} · ${selectedBeat.label}`;
  $('#selectedBeatLabel').textContent=`Reference label: ${selectedBeat.label}`;
  $('#selectedBeatIndex').textContent=`Beat ${selectedBeat.index+1}`;
  renderLargeBeat($('#selectedBeatSvg'),selectedBeat.samples||selectedBeat.points);
  $('#beatDetails').innerHTML=`
    <span>Reference<b>${selectedBeat.label}</b></span>
    <span>Start<b>${selectedBeat.start ?? '—'}</b></span>
    <span>End<b>${selectedBeat.end ?? '—'}</b></span>
    <span>Samples<b>${(selectedBeat.samples||[]).length || 320}</b></span>`;
  $('#predictionResult').innerHTML='<span class="muted-text">Beat selected. Choose a model and run prediction.</span>';

  $('#compareBeatTitle').textContent=`${currentRecord.record_id} · Beat ${selectedBeat.index+1} · Reference ${selectedBeat.label}`;
  $('#compareBeatSubtitle').textContent='Ready for 4-model comparison.';
  $('#compareBeatBadge').textContent=`Beat ${selectedBeat.index+1} · ${selectedBeat.label}`;
  renderLargeBeat($('#compareBeatSvg'),selectedBeat.samples||selectedBeat.points);
  $('#compareBeatDetails').innerHTML=`
    <span>Record<b>${currentRecord.record_id}</b></span>
    <span>Beat<b>${selectedBeat.index+1}</b></span>
    <span>Reference<b>${selectedBeat.label}</b></span>
    <span>Samples<b>${(selectedBeat.samples||[]).length||320}</b></span>`;

  if(autoScroll){
    const box=$(`.beat-box[data-index="${index}"]`);
    if(box) box.scrollIntoView({behavior:'smooth',inline:'center',block:'nearest'});
  }
}

async function loadRecords(){
  currentSplit=$('#splitSelect').value;
  try{
    const data=await getJSON(`/api/dataset/records?split=${encodeURIComponent(currentSplit)}`);
    const sel=$('#recordSelect'); sel.innerHTML='';
    data.records.forEach(r=>{
      const o=document.createElement('option');
      o.value=r.record_id; o.dataset.split=r.split||currentSplit;
      const classes=Object.entries(r.class_counts||{}).filter(([,v])=>v>0).map(([k,v])=>`${k}:${v}`).join(' ');
      o.textContent=`${r.record_id} · ${r.beat_count} beats${classes?' · '+classes:''}`;
      sel.appendChild(o);
    });
    $('#sourceBadge').textContent=data.source;
  }catch(e){ toast(e.message); }
}

async function loadRecord(){
  const sel=$('#recordSelect');
  if(!sel.value){ toast('No ECG sample available'); return; }
  const split=sel.selectedOptions[0]?.dataset.split || $('#splitSelect').value;
  $('#recordMeta').textContent='Loading all ECG beats...';
  try{
    const data=await getJSON(`/api/dataset/record/${encodeURIComponent(split)}/${encodeURIComponent(sel.value)}`);
    currentRecord=data; selectedBeat=null;
    updateClassSelector();
    renderBeatStrip();
    $('#recordMeta').textContent=`${data.record_id} · ${data.source} · all ${data.beat_count} beats loaded`;
    $('#sourceBadge').textContent=data.source;
    if(data.beats.length) selectBeat(0,false);
  }catch(e){ toast(e.message); $('#recordMeta').textContent='Failed to load record.'; }
}

function renderCandidates(){
  const sel=$('#candidateSelect'); sel.innerHTML='';
  candidates.forEach(c=>{
    const o=document.createElement('option'); o.value=c.candidate_id; o.textContent=`${pad4(c.candidate_id)} · ${c.label}`; sel.appendChild(o);
  });
  sel.value=String(selectedCandidateId);
  updateCandidateInfo();
}

function updateCandidateInfo(){
  selectedCandidateId=Number($('#candidateSelect').value);
  const c=candidates.find(x=>x.candidate_id===selectedCandidateId); if(!c) return;
  $('#modelPurpose').textContent=c.purpose;
  $('#mAccuracy').textContent=pct(c.int8_val_accuracy);
  $('#mF1').textContent=f4(c.int8_val_macro_f1);
  $('#mParams').textContent=fmt(c.params);
  $('#mMacs').textContent=fmt(c.macs);
}

function renderPrediction(result){
  const probs=result.probabilities||{};
  $('#predictionResult').innerHTML=`
    <div class="pred-line"><div><small>${result.backend}</small><strong>${result.predicted_class}</strong></div><div><small>Confidence</small><strong>${pct(result.confidence)}</strong></div></div>
    <div class="prob-row">${Object.entries(probs).map(([k,v])=>`<span class="prob-chip ${k===result.predicted_class?'active':''}">${k} ${pct(v)}</span>`).join('')}</div>`;
}

function beatPayload(){
  if(!selectedBeat) throw new Error('Select an ECG beat first.');
  return {candidate_id:selectedCandidateId,record_id:currentRecord?.record_id,beat:selectedBeat};
}

async function runSoftware(){
  try{ const j=await postJSON('/api/predict/software',beatPayload()); renderPrediction(j.result); }
  catch(e){toast(e.message)}
}

function renderFpgaResult(result){
  renderLargeBeat($('#fpgaBeatSvg'),selectedBeat?.samples||selectedBeat?.points||[]);
  const c=candidates.find(x=>x.candidate_id===selectedCandidateId);
  $('#fpgaContextText').textContent=`${currentRecord?.record_id||'—'} · Beat ${(selectedBeat?.index??-1)+1} · Reference ${selectedBeat?.label||'—'}`;
  $('#fpgaModelSummary').textContent=`Candidate ${pad4(c.candidate_id)} · ${c.label} · F1 ${f4(c.int8_val_macro_f1)} · ${fmt(c.params)} params · ${fmt(c.macs)} MACs`;
  $('#fpgaBackendBadge').textContent='PYNQ';
  $('#fpgaPred').textContent=result.predicted_class||'—';
  $('#fpgaConfidence').textContent=`Confidence ${result.confidence!=null?pct(result.confidence):'—'}`;
  $('#fpgaLatency').textContent=result.latency_ms!=null?`${Number(result.latency_ms).toFixed(3)} ms`:'—';
  $('#fpgaThroughput').textContent=result.throughput_inf_s!=null?`${Number(result.throughput_inf_s).toFixed(1)} inf/s`:'—';
  const r=result.resources||{};
  $('#fpgaLut').textContent=r.lut??'—'; $('#fpgaFf').textContent=r.ff??'—'; $('#fpgaDsp').textContent=r.dsp??'—'; $('#fpgaBram').textContent=r.bram??'—';
  $('#fpgaClock').textContent=r.clock_mhz!=null?`${r.clock_mhz} MHz`:'—';
  $('#fpgaNote').textContent=result.note || 'Measured result returned by the PYNQ endpoint.';
}

async function runFpga(){
  try{ const j=await postJSON('/api/predict/fpga',beatPayload()); renderFpgaResult(j.result); setView('fpga'); }
  catch(e){toast(e.message)}
}

function defaultSearchSpace(){
  return {
    kernel_choices:[[3,5,7],[3,5,9],[3,7,9],[5,7,9]],
    channel_choices:[4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64]
  };
}

function setSearchSpaceInputs(searchSpace){
  const ss=searchSpace||defaultSearchSpace();
  const kernelKeys=new Set((ss.kernel_choices||[]).map(x=>x.map(Number).join(',')));
  const channelKeys=new Set((ss.channel_choices||[]).map(Number));
  $$('input[name="kernel_choice"]').forEach(el=>{el.checked=kernelKeys.has(el.value)});
  $$('input[name="channel_choice"]').forEach(el=>{el.checked=channelKeys.has(Number(el.value))});
}

function combination(n,k){
  if(k<0||k>n) return 0;
  k=Math.min(k,n-k);
  let out=1;
  for(let i=1;i<=k;i++) out=out*(n-k+i)/i;
  return Math.round(out);
}

function calculateSearchSpaceSize(ss){
  const k=(ss.kernel_choices||[]).length;
  const c=(ss.channel_choices||[]).length;
  if(!k||!c) return 0;
  return (k**4)*combination(c+3,4);
}

function collectSearchSpace(){
  const kernels=$$('input[name="kernel_choice"]:checked').map(el=>el.value.split(',').map(Number));
  const channels=$$('input[name="channel_choice"]:checked').map(el=>Number(el.value)).sort((a,b)=>a-b);
  return {kernel_choices:kernels,channel_choices:channels};
}

function populateNas(cfg){
  currentConfigSnapshot=JSON.parse(JSON.stringify(cfg));
  const f=$('#nasForm');
  Object.entries(cfg).forEach(([k,v])=>{
    if(k==='search_space') return;
    if(f.elements[k]) f.elements[k].value=Array.isArray(v)?v.join(', '):v;
  });
  setSearchSpaceInputs(cfg.search_space||defaultSearchSpace());
  updateConfigPreview();
}

function collectNas(){
  const f=$('#nasForm'), out={};
  ['population_size','generations','elite_count','tournament_size','epochs_per_candidate','batch_size','early_stop_patience','max_params','max_macs','max_proposal_attempts'].forEach(k=>out[k]=Number(f.elements[k].value));
  ['mutation_rate','crossover_rate','learning_rate','max_memory_mb','f1_tolerance'].forEach(k=>out[k]=Number(f.elements[k].value));
  out.seeds=f.elements.seeds.value;
  out.search_space=collectSearchSpace();
  return out;
}

function updateConfigPreview(){
  try{
    const cfg=collectNas();
    $('#configPreview').textContent=JSON.stringify(cfg,null,2);

    $('#summarySearch').innerHTML=`
      <b>${cfg.population_size}</b> population · <b>${cfg.generations}</b> generations<br>
      Elite <b>${cfg.elite_count}</b> · Tournament <b>${cfg.tournament_size}</b><br>
      Mutation <b>${cfg.mutation_rate}</b> · Crossover <b>${cfg.crossover_rate}</b>`;

    const ss=cfg.search_space;
    const size=calculateSearchSpaceSize(ss);
    const kernelText=ss.kernel_choices.length?ss.kernel_choices.map(x=>`(${x.join(',')})`).join(' · '):'None';
    const channelText=ss.channel_choices.length?`${ss.channel_choices[0]}–${ss.channel_choices.at(-1)} (${ss.channel_choices.length} choices)`:'None';
    $('#summaryArchitecture').innerHTML=`
      Kernels <b>${ss.kernel_choices.length}</b> · Channels <b>${ss.channel_choices.length}</b><br>
      <span title="${kernelText}">${kernelText}</span><br>
      ${channelText} · <b>${fmt(size)}</b> architectures`;
    $('#searchSpaceCount').textContent=fmt(size);
    $('#searchSpaceFormula').textContent=ss.kernel_choices.length&&ss.channel_choices.length
      ? `${ss.kernel_choices.length} kernel choices⁴ × C(${ss.channel_choices.length+3},4) channel tuples`
      : 'Select at least one kernel triplet and one channel width';

    $('#summaryTraining').innerHTML=`
      <b>${cfg.epochs_per_candidate}</b> epochs/candidate · Batch <b>${cfg.batch_size}</b><br>
      LR <b>${cfg.learning_rate}</b> · Patience <b>${cfg.early_stop_patience}</b>`;

    $('#summaryConstraints').innerHTML=`
      Params ≤ <b>${fmt(cfg.max_params)}</b> · MACs ≤ <b>${fmt(cfg.max_macs)}</b><br>
      Memory ≤ <b>${cfg.max_memory_mb} MB</b> · F1 tol <b>${cfg.f1_tolerance}</b><br>
      Proposal attempts <b>${fmt(cfg.max_proposal_attempts)}</b>`;

    const seeds=String(cfg.seeds||'').split(',').map(x=>x.trim()).filter(Boolean);
    $('#summarySeeds').innerHTML=seeds.length?seeds.map(x=>`<span>${x}</span>`).join(''):'<em>No seeds</em>';
  }catch(_e){}
}

async function loadConfigList(selectName=null){
  const j=await getJSON('/api/nas/configs');
  savedConfigs=j.configs||[];

  const dirEl=$('#configDirText');
  if(dirEl) dirEl.textContent=j.config_dir||'PROJECT_ROOT/configs';

  const sel=$('#configSelect');
  sel.innerHTML='';

  if(!savedConfigs.length){
    const o=document.createElement('option');
    o.value='';
    o.textContent='No .json configs found';
    sel.appendChild(o);
    sel.disabled=true;
    currentConfigName='';
    $('#selectedConfigName').textContent='No config';
    $('#selectedConfigFile').textContent='Add a .json file to the project configs folder';
    const pathEl=$('#selectedConfigFilePath');
    if(pathEl) pathEl.textContent=j.config_dir||'';
    $('#configReadonlyBadge').textContent='NO CONFIG';
    $('#runNas').disabled=true;
    return;
  }

  sel.disabled=false;
  savedConfigs.forEach(item=>{
    const o=document.createElement('option');
    o.value=item.name;
    const tags=[];
    if(item.is_original) tags.push('original');
    if(item.is_preferred) tags.push('preferred');
    o.textContent=tags.length ? `${item.name} · ${tags.join(' · ')}` : item.name;
    sel.appendChild(o);
  });

  const fallback=j.preferred_name || savedConfigs[0].name;
  const wanted=
    selectName && savedConfigs.some(x=>x.name===selectName)
      ? selectName
      : (currentConfigName && savedConfigs.some(x=>x.name===currentConfigName)
          ? currentConfigName
          : fallback);

  sel.value=wanted;
  await loadSelectedConfig();
}

async function loadSelectedConfig(){
  const name=$('#configSelect').value;
  if(!name) return;

  currentConfigName=name;
  const j=await getJSON(`/api/nas/config/${encodeURIComponent(currentConfigName)}`);
  populateNas(j.config);

  $('#selectedConfigName').textContent=j.name;
  $('#selectedConfigFile').textContent=j.filename;
  const pathEl=$('#selectedConfigFilePath');
  if(pathEl) pathEl.textContent=j.path||'';

  $('#configReadonlyBadge').textContent=j.is_original?'ORIGINAL · READ-ONLY':'SELECTED CONFIG';
  $('#configReadonlyBadge').className='badge '+(j.is_original?'soft':'');
}

async function saveConfigAsNew(){
  try{
    const payload={name:$('#newConfigName').value.trim(),config:collectNas()};
    const j=await postJSON('/api/nas/config/save',payload);
    $('#newConfigName').value='';
    currentConfigName=j.name;
    await loadConfigList(j.name);
    toast(`Saved ${j.name}.json in the project configs folder.`);
  }catch(e){toast(e.message)}
}

function setNasLamp(state){
  const lamp=$('#nasLamp');
  lamp.className='nas-lamp '+String(state||'IDLE').toLowerCase();
}

async function refreshNasStatus(){
  try{
    const s=await getJSON('/api/nas/status');
    const state=s.state||'IDLE';
    setNasLamp(state);
    $('#nasState').textContent=state;
    $('#nasMessage').textContent=s.message||'—';
    const p=Math.max(0,Math.min(100,Number(s.progress)||0));
    $('#nasProgressBar').style.width=`${p}%`;
    $('#nasProgressText').textContent=`${p.toFixed(1)}%`;
    $('#nasCandidateProgress').textContent=`${s.completed_candidates||0} / ${s.total_candidates||0} candidates`;
    $('#nasSeedText').textContent=`Seed: ${s.active_seed ?? '—'}`;
    $('#nasRunConfig').textContent=s.config_name||'—';
    $('#nasRunOutput').textContent=s.output_dir||'—';
    $('#nasStarted').textContent=s.started_at||'—';
    $('#nasFinished').textContent=s.finished_at||'—';

    const isRunning=state==='RUNNING';
    $('#runNas').disabled=isRunning;
    $('#cancelNas').disabled=!isRunning;
    $('#resetNas').disabled=isRunning;
    $('#configSelect').disabled=isRunning;
    $('#saveConfigAsNew').disabled=isRunning;
  }catch(e){ console.error(e); }
}

async function cancelNas(){
  if(!confirm('Stop the current NAS run? Partial results already written to the run folder will be kept.')) return;
  try{
    const j=await postJSON('/api/nas/cancel',{});
    toast('Cancellation requested.');
    $('#nasState').textContent=j.state||'CANCELLING';
    $('#nasMessage').textContent='Stopping the active NAS process...';
    $('#cancelNas').disabled=true;
    await refreshNasStatus();
  }catch(e){toast(e.message)}
}

async function runNas(){
  if(!currentConfigName){
    toast('Select a config from the project configs folder first.');
    return;
  }
  try{
    const j=await postJSON('/api/nas/run',{
      config_name:currentConfigName,
      config:collectNas()
    });
    toast(`NAS started with the current dashboard values. Snapshot: ${j.output_dir}`);
    await refreshNasStatus();
  }catch(e){toast(e.message)}
}

async function compare(mode){
  if(!selectedBeat){toast('Select an ECG beat first.');return;}
  $('#compareModeBadge').textContent='RUNNING';
  $('#compareTableTitle').textContent=mode==='fpga'?'FPGA comparison':'Software comparison';
  try{
    const j=await postJSON(`/api/compare/${mode}`,{record_id:currentRecord?.record_id,beat:selectedBeat});
    $('#compareModeBadge').textContent=mode==='fpga'?'FPGA':'SOFTWARE';
    $('#compareTable').innerHTML=j.rows.map(row=>{
      const c=row.candidate, r=row.result||{};
      const status=row.ok?'PASS':'ERROR';
      const statusClass=row.ok?'status-pass':'status-error';
      const errorTitle=row.ok?'':` title="${String(row.error||'').replaceAll('"','&quot;')}"`;
      return `<tr>
        <td><span class="role-tag"><span class="role-dot" style="background:${roleColors[c.role]}"></span>${c.label}</span></td>
        <td>${pad4(c.candidate_id)}</td><td>${r.predicted_class??'—'}</td><td>${r.confidence!=null?pct(r.confidence):'—'}</td>
        <td>${f4(c.int8_val_macro_f1)}</td><td>${fmt(c.params)}</td><td>${fmt(c.macs)}</td>
        <td>${r.latency_ms!=null?Number(r.latency_ms).toFixed(3)+' ms':'—'}</td>
        <td>${r.throughput_inf_s!=null?Number(r.throughput_inf_s).toFixed(1):'—'}</td>
        <td class="${statusClass}"${errorTitle}>${status}</td></tr>`;
    }).join('');
  }catch(e){ $('#compareModeBadge').textContent='ERROR'; toast(e.message); }
}

async function refreshStatus(){
  try{
    const s=await getJSON('/api/status');
    $('#datasetMode').textContent=s.dataset.mode;
    $('#datasetPathText').textContent=s.dataset.connected?(s.dataset.csv_dir||'MIT-BIH connected'):'Dataset not found';
    $('#datasetDot').className='dot'+(s.dataset.connected?'':' warn');

    const rt=s.runtime||{};
    $('#runtimeStatus').textContent=`INT8 models ${rt.models_found??0}/${rt.models_total??4} · TensorFlow ${rt.tensorflow_available?'ready':'missing'}`;
    $('#modelDot').className='dot'+((rt.models_found===rt.models_total && rt.tensorflow_available)?'':' warn');

    $('#pynqStatus').textContent=s.pynq_configured?'PYNQ endpoint configured':'PYNQ endpoint not configured';
    $('#pynqDot').className='dot'+(s.pynq_configured?'':' muted');

    if($('#nasBackendStatus')){
      const ready=Boolean(s.nas?.search_space_backend_ready);
      $('#nasBackendStatus').textContent=ready?'Search-space backend ready':'Search-space backend patch required';
      $('#nasBackendDot').className='dot'+(ready?'':' muted');
    }
  }catch(e){toast(e.message)}
}

function bind(){
  $$('.nav-btn').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.view)));
  $('#splitSelect').addEventListener('change',async()=>{await loadRecords();await loadRecord();});
  $('#loadRecord').addEventListener('click',loadRecord);
  $('#classSelect').addEventListener('change',()=>{classCursor=-1;applyClassVisuals();});
  $('#findClass').addEventListener('click',()=>findClass(true));
  $('#findNext').addEventListener('click',()=>findClass(false));
  $('#candidateSelect').addEventListener('change',updateCandidateInfo);
  $('#runSoftware').addEventListener('click',runSoftware);
  $('#runFpga').addEventListener('click',runFpga);
  $('#rerunFpga').addEventListener('click',runFpga);
  $('#backToPrediction').addEventListener('click',()=>setView('prediction'));

  $('#configSelect').addEventListener('change',loadSelectedConfig);
  $('#saveConfigAsNew').addEventListener('click',saveConfigAsNew);
  $('#resetNas').addEventListener('click',loadSelectedConfig);
  $('#runNas').addEventListener('click',runNas);
  $('#cancelNas').addEventListener('click',cancelNas);
  $('#nasForm').addEventListener('input',updateConfigPreview);
  $$('.search-space-card input[type="checkbox"]').forEach(el=>el.addEventListener('change',updateConfigPreview));

  $('#compareSoftware').addEventListener('click',()=>compare('software'));
  $('#compareFpga').addEventListener('click',()=>compare('fpga'));

  $('#beatScrollRange').addEventListener('input',e=>{$('#beatScroll').scrollLeft=Number(e.target.value)});
  $('#beatScroll').addEventListener('scroll',()=>{$('#beatScrollRange').value=$('#beatScroll').scrollLeft});
}

async function init(){
  bind();
  candidates=await getJSON('/api/candidates');
  renderCandidates();
  await refreshStatus();
  await loadRecords();
  await loadRecord();
  await loadConfigList();
  await refreshNasStatus();
  nasPollTimer=setInterval(refreshNasStatus,1500);
}

init().catch(e=>{console.error(e);toast('Dashboard init failed: '+e.message)});
