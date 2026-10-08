(() => {
  'use strict';
  const el = id => document.getElementById(id);
  const number = id => Number(el(id).value);
  const format = value => value == null ? '—' : Number(value).toLocaleString('en-DK', {maximumFractionDigits: 2});
  const fields = ['labFormula','labName','labSignal','labForecast','labLookback','labDirection','labLower','labUpper','labEvaluation','labSetup','labCapacity','labPower','labEfficiency','labChargeFee','labSellFee','labCapital','labPosition','labCost','labLoss'];
  let fundamentals = null;
  let initialReplay = true;
  let replayCursor = 0, replayTimer = null, runSerial = 0;
  const runs = [], runColors = ['#4fe0b0','#a5b4fc','#f5b54a','#ff8fab','#7dd3fc'];
  const chartFont = '11px ui-monospace,SFMono-Regular,Consolas,monospace';
  // Saved model inputs shown as tiles above the replay chart: [key in result.factors, label, unit].
  const factorTiles = [['wind','Wind','MW'],['solar','Solar','MW'],['demand','Demand','MW'],['forecast','Price forecast','DKK'],
    ['baseline','Hourly baseline','DKK'],['temperature','Temperature · 1d lag',''],['gas','Gas price · 1d lag',''],['rank','Forecast daily rank','/100']];
  // Strategy library. Thresholds sit near the 20th/80th percentiles of each signal in the saved DK1 data.
  // buy_low: long when signal <= lower, short when signal >= upper; buy_high is the mirror image.
  const presets = [
    {tag:'Daily shape',name:'Cheapest 20% / priciest 20%',desc:'Long the day’s 20% lowest-forecast intervals, short the 20% highest.',formula:'signal = forecast_rank',direction:'buy_low',lower:20,upper:80},
    {tag:'Daily shape',name:'Extremes only · 10% / 10%',desc:'Trade only the day’s most extreme forecast prices.',formula:'signal = forecast_rank',direction:'buy_low',lower:10,upper:90},
    {tag:'Daily shape',name:'Wide-spread days only',desc:'Cheapest/priciest 20%, but only on days whose forecast spread exceeds 600 DKK.',formula:'signal = 50 + (forecast_rank - 50) * clamp((forecast_spread - 600) * 1000, 0, 1)',direction:'buy_low',lower:20,upper:80},
    {tag:'Daily shape',name:'Daily z-score reversion',desc:'Long when the forecast is 1σ below its daily mean, short 1σ above.',formula:'signal = forecast_z',direction:'buy_low',lower:-1,upper:1},
    {tag:'Time of day',name:'Night long, evening short',desc:'Long 00:00–05:00, short from 17:00, Copenhagen time.',formula:'signal = hour',direction:'buy_low',lower:5,upper:17},
    {tag:'Fundamentals',name:'Residual demand',desc:'Long when demand minus wind and solar is tight, short when it is loose.',formula:'signal = demand - wind - solar',direction:'buy_high',lower:600,upper:2300},
    {tag:'Fundamentals',name:'Wind fade',desc:'Long on calm intervals (wind < 700 MW), short when wind > 2,300 MW.',formula:'signal = -wind',direction:'buy_high',lower:-2300,upper:-700},
    {tag:'Fundamentals',name:'Power vs gas',desc:'Long when power is cheap relative to yesterday’s gas price, short when rich.',formula:'signal = forecast / gas_price_lag_96',direction:'buy_low',lower:18,upper:29},
    {tag:'Forecast',name:'Below the hourly baseline',desc:'Long when the 15-min forecast sits under the hourly baseline, short when well above.',formula:'signal = forecast - baseline',direction:'buy_low',lower:5,upper:70},
    {tag:'Forecast',name:'Forecast momentum',desc:'Follow the forecast’s 1-hour move: long if up > 65 DKK, short if down > 65.',signal:'forecast_change',lookback:4,direction:'buy_high',lower:-65,upper:65},
  ];
  function presetRule(p) {
    const low=p.direction==='buy_low';
    return `${(p.formula||'signal = forecast change, 1h').replace(/^signal = /,'')} · long ${low?'≤':'≥'} ${low?p.lower:p.upper} · short ${low?'≥':'≤'} ${low?p.upper:p.lower}`;
  }
  function buildLibrary() {
    el('strategyLibrary').replaceChildren(...presets.map((p,i)=>{
      const card=document.createElement('button');card.type='button';card.className='strategy-card';card.dataset.preset=String(i);card.setAttribute('aria-pressed','false');
      const parts=[['span','tag',p.tag],['b','',p.name],['small','',p.desc],['code','',presetRule(p)]];
      for(const [tag,cls,text] of parts){const node=document.createElement(tag);if(cls)node.className=cls;node.textContent=text;card.append(node);}
      card.addEventListener('click',()=>applyPreset(p));
      return card;
    }));
  }
  function matchesPreset(p) {
    const signal=p.signal||'formula';
    return el('labSignal').value===signal && (signal!=='formula'||el('labFormula').value.trim()===p.formula)
      && (signal!=='forecast_change'||number('labLookback')===p.lookback)
      && el('labDirection').value===p.direction && number('labLower')===p.lower && number('labUpper')===p.upper;
  }
  function applyPreset(p) {
    if(busy||readingFile)return;
    el('labSignal').value=p.signal||'formula';
    if(p.formula)el('labFormula').value=p.formula;
    if(p.lookback)el('labLookback').value=p.lookback;
    el('labName').value=p.name; el('labDirection').value=p.direction; el('labLower').value=p.lower; el('labUpper').value=p.upper;
    updateFields(); dirty(); initialReplay=true; el('labForm').requestSubmit();
  }
  function compact(value) {return value==null||!Number.isFinite(value)?'—':Number(value).toLocaleString('en-DK',{maximumFractionDigits:Math.abs(value)>=100?0:2});}
  function buildFactors() {
    el('factorGrid').replaceChildren(...factorTiles.map(([key,label,unit])=>{
      const tile=document.createElement('div');tile.className='factor';tile.dataset.factor=key;
      const name=document.createElement('span');name.textContent=label;
      const value=document.createElement('strong');value.textContent='—';
      if(unit){const small=document.createElement('small');small.textContent=unit;value.append(small);}
      const delta=document.createElement('i');delta.className='flat';delta.textContent='';
      tile.append(name,value,delta);return tile;
    }));
  }
  function renderFactors() {
    const i=replayCursor-1;
    for(const tile of el('factorGrid').children){
      const series=result?.factors?.[tile.dataset.factor], value=i>=0?series?.[i]:null, previous=i>0?series?.[i-1]:null;
      tile.querySelector('strong').firstChild.textContent=compact(value)+(tile.querySelector('small')?' ':'');
      const delta=tile.querySelector('i'), change=value!=null&&previous!=null?value-previous:null;
      delta.className=change>0?'up':change<0?'down':'flat';
      delta.textContent=change==null?'':change===0?'±0':`${change>0?'▲':'▼'} ${compact(Math.abs(change))}`;
      delta.title=change==null?'':'Change since the previous 15-minute interval';
    }
  }
  // Start the replay once the panel is on screen (it sits below the fold on the landing page).
  function playWhenVisible() {
    const start=()=>{if(result&&!replayTimer&&replayCursor===0)el('replayPlay').click();};
    if(!('IntersectionObserver' in window)){start();return;}
    const observer=new IntersectionObserver(entries=>{if(entries.some(entry=>entry.isIntersecting)){observer.disconnect();start();}},{rootMargin:'0px 0px -40% 0px'});
    observer.observe(el('replayWorkspace'));
  }
  function pauseReplay() {clearInterval(replayTimer);replayTimer=null;el('replayPlay').textContent='Play';}
  function addRun(data) {
    pauseReplay();
    const signature=data.intervals.map(row=>row.HourUTC).join('|');
    if(runs.length && runs[0].signature!==signature) runs.length=0;
    data.runId=String(++runSerial); data.signature=signature;
    runs.push(data);if(runs.length>5)runs.shift();
    result=data;replayCursor=0;
    el('replayRun').replaceChildren();
    for(const run of runs){const option=document.createElement('option');option.value=run.runId;option.textContent=`#${run.runId} ${run.name}: ${run.settings.signal==='formula'?run.settings.formula:run.settings.signal}`;el('replayRun').append(option);}
    el('replayRun').value=data.runId;
  }
  function positionName(value) {return value>0?'Long':value<0?'Short':'Flat';}
  function renderReplay() {
    el('replayCursor').max=result.intervals.length;el('replayCursor').value=replayCursor;
    const row=result.intervals[replayCursor-1];
    el('replayTime').textContent=row?`${row.HourUTC.slice(0,16).replace('T',' ')} UTC · interval ${replayCursor.toLocaleString()} / ${result.intervals.length.toLocaleString()}`:'Start of test · no intervals completed';
    el('replayPrice').textContent=row?compact(row.Actual_Price):'—';
    renderFactors();
    const strength=row?ReplayMath.strength(row.Custom_Signal,result.settings):null;
    el('replaySignal').textContent=row?format(row.Custom_Signal):'-';
    el('replayStrength').textContent=strength==null?'-':`${format(strength)}%`;
    el('replayStrength').className=strength>=100?'positive':strength<=-100?'negative':'';
    el('replayPosition').textContent=positionName(row?.Position||0);
    el('replayPosition').className=row?.Position>0?'positive':row?.Position<0?'negative':'';
    el('replayExposure').textContent=row?`${format(row.Position_MWh)} MWh`:'0 MWh';
    el('replayAfter').textContent=positionName(row?.Position_After_Settlement||0);
    const netCashflow=row?.Cumulative_Cashflow||0;
    el('replayNetCashflow').textContent=format(netCashflow);
    el('replayNetCashflow').className=netCashflow>0?'positive':netCashflow<0?'negative':'';
    const body=el('replayComparisons');body.replaceChildren();
    runs.forEach((run,i)=>{
      const interval=run.intervals[replayCursor-1], strength=interval?ReplayMath.strength(interval.Custom_Signal,run.settings):null;
      const tr=document.createElement('tr');
      for(const value of [`#${run.runId} ${run.name}`,interval?format(interval.Custom_Signal):'-',strength==null?'-':`${format(strength)}%`,positionName(interval?.Position||0),format(interval?.Cumulative_Cashflow||0)]){
        const td=document.createElement('td');td.textContent=value;tr.append(td);
      }
      tr.firstChild.style.color=runColors[i];body.append(tr);
    });
  }
  let records = null, result = null, busy = false, readingFile = false;
  function status(text, error = false) { el('labStatus').textContent = text; el('labStatus').className = error ? 'error' : ''; }
  function dirty() {
    pauseReplay();
    if (!result) return;
    el('labResults').classList.add('stale'); el('labExport').disabled = true;
    status('Rules changed. Run again to update the results.');
  }
  function updateFields() {
    const csv = el('labSignal').value === 'csv', change = el('labSignal').value === 'forecast_change';
    el('labSetup').value='prop';
    const prop = true;
    const formula = el('labSignal').value === 'formula';
    el('labFormulaFields').hidden = !formula; el('labFormulaExtras').hidden = !formula;
    el('labSourceNote').hidden = formula;
    el('labSourceNote').textContent = formula ? '' : `Signal source: ${el('labSignal').selectedOptions[0].textContent}. Switch to “Write a formula” in Strategy settings to code your own.`;
    el('labCsvFields').hidden = !csv; el('labForecastFields').hidden = csv;
    el('labLookbackFields').hidden = !change;
    el('labBatteryFields').hidden = prop; el('labPropFields').hidden = !prop;
    for (const id of ['labBatteryFields','labPropFields','labForecastFields','labLookbackFields','labCsvFields','labFormulaFields','labFormulaExtras']) {
      el(id).querySelectorAll('input,select,textarea,button').forEach(input => { input.disabled = busy || el(id).hidden; });
    }
    const buy = prop ? 'long' : 'charge', sell = prop ? 'short' : 'discharge';
    const low = el('labDirection').value === 'buy_low';
    el('labRuleSummary').textContent = `position = ${low ? buy : sell} if signal ≤ ${el('labLower').value} · ${low ? sell : buy} if signal ≥ ${el('labUpper').value} · else ${prop ? 'flat' : 'hold'}`;
    el('strategyLibrary').querySelectorAll('[data-preset]').forEach(card => card.setAttribute('aria-pressed', String(matchesPreset(presets[Number(card.dataset.preset)]))));
    const help = {
      formula: 'Your formula is calculated at each timestamp. Set the direction and thresholds to turn the signal into positions.',
      baseline_spread: 'Forecast minus the hourly baseline, in DKK/MWh. Negative values mean the forecast is lower.',
      forecast: 'The forecast price in DKK/MWh. Set thresholds in the same units.',
      forecast_change: 'Current forecast minus the forecast that many intervals earlier, in DKK/MWh. Missing warm-up history produces a hold.',
      csv: 'Thresholds use your signal’s units. Only upload values you could have known at decision time; the tool cannot verify their provenance.',
    };
    el('labSignalHelp').textContent = help[el('labSignal').value];
  }
  function download(name, text) {
    const url = URL.createObjectURL(new Blob([text], {type:'text/csv;charset=utf-8'}));
    const link = document.createElement('a'); link.href = url; link.download = name; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function parseCsv(text, fundamental = false) {
    // Quoted fields and CRLF are accepted; exactly two named columns are required.
    const rows = []; let row = [], cell = '', quoted = false;
    text = text.replace(/^\uFEFF/, '');
    for (let i=0; i<text.length; i++) {
      const char = text[i];
      if (char === '"') {
        if (quoted && text[i+1] === '"') { cell += '"'; i++; } else quoted = !quoted;
      } else if (!quoted && (char === ',' || char === '\n')) {
        row.push(cell.trim()); cell = '';
        if (char === '\n') { if (row.some(value => value !== '')) rows.push(row); row = []; }
      } else cell += char;
    }
    if (quoted) throw new Error('CSV contains an unclosed quote.');
    row.push(cell.trim()); if (row.some(value => value !== '')) rows.push(row);
    const header = rows.shift() || [], time = header.indexOf('HourUTC'), signal = header.indexOf('Signal');
    if (!fundamental && (header.length !== 2 || time < 0 || signal < 0)) throw new Error('CSV needs exactly two headers: HourUTC,Signal.');
    if (fundamental && (time < 0 || header.length < 2 || new Set(header).size !== header.length || header.some(key => !['HourUTC','demand','outages','wind','solar'].includes(key)))) throw new Error('Use HourUTC plus demand, outages, wind and/or solar columns.');
    if (!rows.length || rows.length > 20000) throw new Error('CSV must contain 1–20,000 data rows.');
    const seen = new Set();
    return rows.map((values, i) => {
      const stamp = values[time], value = values[signal];
      if (values.length !== header.length || !stamp || (!fundamental && (!value || !Number.isFinite(Number(value))))) throw new Error(`Invalid timestamp or numeric signal on CSV row ${i+2}.`);
      if (!/(Z|[+-]\d{2}:\d{2})$/i.test(stamp) || !Number.isFinite(Date.parse(stamp))) throw new Error(`CSV row ${i+2} needs an ISO timestamp with a timezone.`);
      const canonical = new Date(stamp).toISOString();
      if (seen.has(canonical)) throw new Error(`Duplicate timestamp on CSV row ${i+2}.`);
      seen.add(canonical);
      if (fundamental) {
        const row = {HourUTC:canonical};
        header.forEach((key,j) => {
          if (key === 'HourUTC') return;
          if (!values[j] || !Number.isFinite(Number(values[j]))) throw new Error(`CSV row ${i+2}: ${key} must be numeric; fill every included column.`);
          row[key] = Number(values[j]);
        });
        return row;
      }
      return {HourUTC:canonical, Signal:Number(value)};
    });
  }
  function request() {
    if (number('labLower') >= number('labUpper')) throw new Error('Lower threshold must be below upper threshold.');
    if (el('labSignal').value === 'csv' && !records) throw new Error('Upload a valid signal CSV first.');
    const efficiency = Math.sqrt(number('labEfficiency') / 100);
    return {
      name:el('labName').value.trim() || 'Custom strategy', trading_setup:el('labSetup').value,
      forecast_col:el('labForecast').value, signal:el('labSignal').value,
      formula:el('labSignal').value === 'formula' ? el('labFormula').value : 'signal = -wind', fundamental_records:el('labSignal').value === 'formula' ? fundamentals : null,
      lower:number('labLower'), upper:number('labUpper'), direction:el('labDirection').value,
      lookback:el('labSignal').value === 'forecast_change' ? number('labLookback') : 4, evaluation:el('labEvaluation').value,
      battery:el('labSetup').value === 'battery' ? {capacity_mwh:number('labCapacity'), power_mw:number('labPower'), initial_soc_mwh:0,
        charge_efficiency:efficiency, discharge_efficiency:efficiency,
        charge_fee_per_mwh:number('labChargeFee'), discharge_fee_per_mwh:number('labSellFee')} : undefined,
      prop:el('labSetup').value === 'prop' ? {initial_capital_dkk:number('labCapital'), position_size_mwh:number('labPosition'),
        transaction_cost_dkk_per_mwh:number('labCost'), max_daily_loss_dkk:number('labLoss') || null} : undefined,
      signal_records:el('labSignal').value === 'csv' ? records : null,
    };
  }
  function draw() {
    if (!result) return;
    const canvas = el('labChart'), rect = canvas.getBoundingClientRect(), scale = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * scale); canvas.height = Math.round(rect.height * scale);
    const ctx = canvas.getContext('2d'); ctx.scale(scale,scale);
    const series=runs.map(run=>[0,...run.intervals.slice(0,replayCursor).map(row=>row.Cumulative_Cashflow)]);
    const values=series.flat();
    // Compact sparkline: min/max labels on the right, dashed zero line when P&L crosses zero.
    const low = Math.min(...values), high = Math.max(...values), pad = Math.max((high-low)*.1,1);
    const min = low-pad, max = high+pad, left = 0, right = rect.width-64, top = 6, bottom = rect.height-6;
    const y = value => bottom-(value-min)/(max-min)*(bottom-top);
    ctx.font = chartFont; ctx.lineWidth = 1; ctx.textAlign = 'left';
    if (min < 0 && max > 0) { ctx.setLineDash([3,4]); ctx.strokeStyle='#33416f'; ctx.beginPath(); ctx.moveTo(left,y(0)); ctx.lineTo(right,y(0)); ctx.stroke(); ctx.setLineDash([]); }
    const active = runs.indexOf(result);
    series.forEach((points,j)=>{
      ctx.strokeStyle=runColors[j];ctx.lineWidth=j===active?2.2:1.4;ctx.globalAlpha=j===active?1:.6;ctx.beginPath();
      points.forEach((value,i)=>{const x=left+i/result.intervals.length*(right-left);if(i===0)ctx.moveTo(x,y(value));else ctx.lineTo(x,y(value));});ctx.stroke();
    });
    ctx.globalAlpha=1; ctx.fillStyle='#9aa8cc';
    ctx.fillText(compact(high),right+8,top+8); ctx.fillText(compact(low),right+8,bottom-2);
    drawSignal();
  }
  function drawSignal(pointer = null) {
    if (!result || el('replayWorkspace').hidden) return;
    const canvas=el('signalCanvas'), rect=canvas.getBoundingClientRect(), scale=window.devicePixelRatio||1;
    if (!rect.width) return;
    canvas.width=Math.round(rect.width*scale);canvas.height=Math.round(rect.height*scale);
    const ctx=canvas.getContext('2d');ctx.scale(scale,scale);ctx.font=chartFont;
    const start=Math.max(0,replayCursor-96), rows=result.intervals.slice(start,replayCursor);
    const left=65,right=rect.width-16,width=right-left,top=16,priceBottom=rect.height*.55,signalTop=rect.height*.67,bottom=rect.height-28;
    ctx.fillStyle='#9aa8cc';ctx.fillText('SIGNAL · formula output units',left,signalTop-14);
    if (!rows.length) {
      ctx.fillText('Press Play to reveal',left,65);
      ctx.fillText('completed intervals.',left,83);
      el('signalChartStatus').textContent='No intervals completed yet. Use the replay controls below.';
      return;
    }
    const extent=values=>{const lo=Math.min(...values),hi=Math.max(...values),pad=Math.max((hi-lo)*.12,1);return [lo-pad,hi+pad];};
    const [priceMin,priceMax]=extent(rows.map(row=>row.Actual_Price));
    const signals=rows.map(row=>row.Custom_Signal).filter(Number.isFinite);
    const [signalMin,signalMax]=extent([...signals,result.settings.lower,result.settings.upper]);
    const x=i=>left+(rows.length===1?width/2:i/(rows.length-1)*width);
    const priceY=value=>priceBottom-(value-priceMin)/(priceMax-priceMin)*(priceBottom-top);
    const signalY=value=>bottom-(value-signalMin)/(signalMax-signalMin)*(bottom-signalTop);
    rows.forEach((row,i)=>{
      ctx.fillStyle=row.Position>0?'rgba(79,224,176,.13)':row.Position<0?'rgba(255,143,171,.13)':'rgba(255,255,255,0)';
      const band=width/rows.length;ctx.fillRect(left+i*band,top,band,priceBottom-top);ctx.fillRect(left+i*band,signalTop,band,bottom-signalTop);
    });
    ctx.strokeStyle='#243159';ctx.lineWidth=1;
    for(let i=0;i<=3;i++){
      const value=priceMin+(priceMax-priceMin)*i/3,y=priceY(value);
      ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.textAlign='right';ctx.fillStyle='#9aa8cc';ctx.fillText(format(value),left-9,y+4);
    }
    for(const threshold of [result.settings.lower,result.settings.upper]){
      ctx.setLineDash([4,5]);ctx.strokeStyle='#8193bf';ctx.beginPath();ctx.moveTo(left,signalY(threshold));ctx.lineTo(right,signalY(threshold));ctx.stroke();ctx.setLineDash([]);
      ctx.textAlign='right';ctx.fillText(format(threshold),left-9,signalY(threshold)+4);
    }
    function line(key,y,color){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();let connected=false;rows.forEach((row,i)=>{if(!Number.isFinite(row[key])){connected=false;return;}if(connected)ctx.lineTo(x(i),y(row[key]));else ctx.moveTo(x(i),y(row[key]));connected=true;});ctx.stroke();}
    line('Actual_Price',priceY,'#a5b4fc');line('Custom_Signal',signalY,'#f5b54a');
    const inspected=pointer==null?rows.length-1:Math.round(Math.max(0,Math.min(1,pointer))*(rows.length-1));
    const row=rows[inspected];ctx.setLineDash([3,4]);ctx.strokeStyle='#c9d2ee';ctx.beginPath();ctx.moveTo(x(inspected),top);ctx.lineTo(x(inspected),bottom);ctx.stroke();ctx.setLineDash([]);
    ctx.fillStyle='#a5b4fc';ctx.beginPath();ctx.arc(x(inspected),priceY(row.Actual_Price),3,0,Math.PI*2);ctx.fill();
    ctx.fillStyle='#9aa8cc';ctx.textAlign='left';ctx.fillText(rows[0].HourUTC.slice(5,16).replace('T',' '),left,rect.height-6);ctx.textAlign='right';ctx.fillText(rows.at(-1).HourUTC.slice(5,16).replace('T',' ')+' UTC',right,rect.height-6);
    el('signalChartStatus').textContent=`${row.HourUTC.slice(0,16).replace('T',' ')} UTC · price ${format(row.Actual_Price)} DKK/MWh · signal ${format(row.Custom_Signal)} · ${positionName(row.Position||0)} · last ${rows.length} completed intervals`;
  }
  function render() {
    const shown=result.intervals.slice(0,replayCursor);
    const s=ReplayMath.stats(shown,result.settings.prop.initial_capital_dkk), prop=true;
    renderReplay();
    el('replayWorkspace').hidden=false;
    el('labResults').hidden=false; el('labResults').classList.remove('stale'); el('labExport').disabled=false;
    el('labResultTitle').textContent=result.name;
    el('activeFormula').textContent=result.settings.signal==='formula'?result.settings.formula:result.settings.signal.replaceAll('_',' ');
    const longHigh=result.settings.direction==='buy_high';
    el('activeRule').textContent=`${longHigh?'Short':'Long'} ≤ ${format(result.settings.lower)} · ${longHigh?'Long':'Short'} ≥ ${format(result.settings.upper)} · Flat between thresholds`;
    el('labPeriod').textContent=`${prop ? 'Prop proxy' : 'Physical battery'} · ${result.period.start.slice(0,16)} to ${result.period.end.slice(0,16)} UTC · ${result.period.rows.toLocaleString()} intervals · ${result.settings.signal === 'formula' ? result.settings.formula : result.settings.signal} · thresholds ${result.settings.lower} / ${result.settings.upper}`;
    el('labPnl').textContent=format(s.total_cashflow); el('labPnl').className=s.total_cashflow>=0?'positive':'negative';
    el('labDrawdown').textContent=format(s.max_drawdown); el('labActive').textContent=format(s.active_intervals);
    el('labWarmup').textContent=`${s.warmup_intervals} warm-up holds · fees ${format(s.total_fee_cost)} DKK`;
    el('labLastLabel').textContent=prop?'Ending equity · DKK':'Final stored energy · MWh';
    el('labLast').textContent=format(prop?s.ending_equity_dkk:s.final_soc_mwh);
    el('labLastNote').textContent='Through completed replay intervals';
    const body=el('labRows'), otherBody=el('labOtherRows'); body.replaceChildren(); otherBody.replaceChildren();
    const fragment=document.createDocumentFragment(), otherFragment=document.createDocumentFragment();
    let visibleCount=0, otherCount=0;

    for (const row of shown) {
      const tr=document.createElement('tr');
      const requested = prop ? ({charge:'long',discharge:'short',hold:'flat'}[row.Signal_Action] || row.Signal_Action) : row.Signal_Action;
      for(const value of [row.HourUTC.replace('T',' ').replace('.000Z',''),format(row.Custom_Signal),requested,row.Action,format(row.Cashflow),format(row.Cumulative_Cashflow)]) {
        const td=document.createElement('td'); td.textContent=value;
        if (tr.children.length === 2 || tr.children.length === 3) {
          if (value === 'charge' || value === 'short' || value === 'imbalance-close-short') td.className='negative';
          if (value === 'discharge' || value === 'long' || value === 'imbalance-close-long') td.className='positive';
        }
        tr.append(td);
      }
      const executed = prop ? ['long','short','exit','imbalance-close-long','imbalance-close-short'].includes(row.Action) : ['charge','discharge'].includes(row.Action) && Math.abs(row.Dispatch_MW)>0;
      if (executed) {fragment.append(tr); visibleCount++;} else {otherFragment.append(tr); otherCount++;}
    }
    body.append(fragment); otherBody.append(otherFragment);
    el('labLogCount').textContent=`${visibleCount} executed intervals.`;
    el('labOtherCount').textContent=`Show other intervals (${otherCount})`;
    el('labOtherIntervals').hidden=otherCount===0;
    el('labNoTrades').hidden=visibleCount!==0;
    draw();
  }
  el('labForm').addEventListener('submit',async event=>{
    event.preventDefault(); if(busy || readingFile)return;
    let payload; try{payload=request();}catch(error){status(error.message,true);return;}
    let autoplay=false; pauseReplay(); busy=true; el('labForm').setAttribute('aria-busy','true');
    el('labForm').querySelectorAll('input,select,textarea,button').forEach(input=>{input.disabled=true;});
    status('Running backtest…'); el('labRun').textContent='Running…';
    try {
      const response=await fetch('/api/custom-strategy',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),signal:AbortSignal.timeout(60000)});
      const data=await response.json();
      if(!response.ok)throw new Error(Array.isArray(data.detail)?data.detail.map(item=>item.msg).join(' '):(data.detail || `Backtest failed (${response.status}).`));
      addRun(data); render(); status('Compiled — the shaded columns are where the rule holds a position.');
      el('replayLaunchStatus').hidden=true;
      if(initialReplay){initialReplay=false;autoplay=true;}
    }catch(error){status(error.message,true);if(!result){el('replayLaunchStatus').hidden=false;el('replayLaunchStatus').textContent='Replay could not load: '+error.message+' Adjust the formula below and press Run backtest.';}initialReplay=false;}
    finally{busy=false;el('labForm').removeAttribute('aria-busy');el('labRun').textContent='Run backtest';el('labForm').querySelectorAll('input,select,textarea,button').forEach(input=>{input.disabled=false;});updateFields();if(autoplay)playWhenVisible();}
  });
  fields.forEach(id=>el(id).addEventListener('input',()=>{updateFields();dirty();}));
  el('labCsv').addEventListener('change',async()=>{
    records=null; dirty(); const file=el('labCsv').files[0];
    if(!file){el('labCsvStatus').textContent='No file loaded.';return;}
    readingFile=true;el('labRun').disabled=true;
    try{if(file.size>2000000)throw new Error('CSV must be smaller than 2 MB.');records=parseCsv(await file.text());el('labCsvStatus').textContent=`${records.length.toLocaleString()} signal rows loaded.`;status('CSV loaded. Run the backtest to validate timestamp coverage.');}
    catch(error){el('labCsvStatus').textContent=error.message;status(error.message,true);}
    finally{readingFile=false;el('labRun').disabled=false;}
  });
  el('labTemplate').addEventListener('click',async()=>{
    try{const response=await fetch('/api/results');if(!response.ok)throw new Error('Could not load the saved timestamps.');const data=await response.json();download('signal-template.csv','HourUTC,Signal\n'+data.prices.map(row=>`${row.HourUTC},0`).join('\n'));status('Template downloaded. Replace the zero signals with your values.');}
    catch(error){status(error.message,true);}
  });
  el('labExport').addEventListener('click',()=>{
    if(!result)return; const keys=Object.keys(result.intervals[0]);
    const quote=value=>'"'+String(value??'').replaceAll('"','""')+'"';
    download('custom-strategy-results.csv',keys.join(',')+'\n'+result.intervals.map(row=>keys.map(key=>quote(row[key])).join(',')).join('\n'));
  });
  el('labSave').addEventListener('click',()=>{
    try{localStorage.setItem('power-trader-custom-rules-v1',JSON.stringify(Object.fromEntries(fields.map(id=>[id,el(id).value]))));status('Rules saved on this device. CSV files are not saved.');}catch{status('Browser storage is unavailable.',true);}
  });
  el('labLoad').addEventListener('click',()=>{
    try{const saved=JSON.parse(localStorage.getItem('power-trader-custom-rules-v1'));if(!saved)throw new Error('No saved rules on this device.');for(const id of fields)if(typeof saved[id]==='string')el(id).value=saved[id];fundamentals=null;el('labFundamentals').value='';el('labFundamentalStatus').textContent='Upload fundamentals again if your formula needs them.';records=null;el('labCsv').value='';el('labCsvStatus').textContent='No file loaded.';updateFields();dirty();status('Rules loaded. Upload your CSV again if needed, then run the test.');}catch(error){status(error.message,true);}
  });
  el('labFundamentals').addEventListener('change', async () => {
    fundamentals = null; dirty(); const file = el('labFundamentals').files[0];
    if (!file) {el('labFundamentalStatus').textContent='Using saved forecasts only.'; return;}
    readingFile=true; el('labRun').disabled=true;
    try {
      if(file.size>2000000) throw new Error('CSV must be smaller than 2 MB.');
      fundamentals=parseCsv(await file.text(),true);
      el('labFundamentalStatus').textContent=`${fundamentals.length.toLocaleString()} rows loaded: ${Object.keys(fundamentals[0]).filter(key=>key!=='HourUTC').join(', ')}.`;
      status('Fundamentals loaded. Run to check timestamp coverage.');
    } catch(error) {el('labFundamentalStatus').textContent=error.message;status(error.message,true);}
    finally {readingFile=false;el('labRun').disabled=false;}
  });
  el('labFundamentalTemplate').addEventListener('click', async () => {
    try {
      const response=await fetch('/api/results');if(!response.ok)throw new Error('Could not load timestamps.');
      const data=await response.json();
      download('fundamentals-template.csv','HourUTC,outages\n'+data.prices.map(row=>`${row.HourUTC},`).join('\n'));
      status('Template downloaded. Fill outages in MW; blank cells are not zero.');
    } catch(error) {status(error.message,true);}
  });
  document.querySelectorAll('[data-variable]').forEach(button => button.addEventListener('click', () => {
    const editor=el('labFormula');
    editor.setRangeText(button.dataset.variable,editor.selectionStart,editor.selectionEnd,'end');
    editor.focus(); dirty();
  }));
  el('replayPlay').addEventListener('click',()=>{
    if(!result)return;
    if(replayTimer){pauseReplay();return;}
    if(replayCursor===result.intervals.length){replayCursor=0;render();}
    el('replayPlay').textContent='Pause';
    replayTimer=setInterval(()=>{replayCursor=Math.min(result.intervals.length,replayCursor+Number(el('replaySpeed').value));render();if(replayCursor===result.intervals.length)pauseReplay();},1000);
  });
  el('replayStep').addEventListener('click',()=>{if(!result)return;pauseReplay();replayCursor=Math.min(result.intervals.length,replayCursor+1);render();});
  el('replayReset').addEventListener('click',()=>{if(!result)return;pauseReplay();replayCursor=0;render();});
  el('replayEnd').addEventListener('click',()=>{if(!result)return;pauseReplay();replayCursor=result.intervals.length;render();});
  el('replayCursor').addEventListener('input',()=>{if(!result)return;pauseReplay();replayCursor=Number(el('replayCursor').value);render();});
  el('replayRun').addEventListener('change',()=>{if(!runs.length)return;pauseReplay();result=runs.find(run=>run.runId===el('replayRun').value);render();status('Viewing a completed run. Edit the form and run again to add a comparison.');});
  el('latestLoad').addEventListener('click',async()=>{
    el('latestLoad').disabled=true;el('latestStatus').textContent='Fetching published DK1 prices...';
    try{
      const response=await fetch('/api/latest-dk1-prices',{signal:AbortSignal.timeout(25000)}), data=await response.json();
      if(!response.ok)throw new Error(data.detail||'Price service unavailable.');
      el('latestRows').replaceChildren();
      const dates=new Intl.DateTimeFormat('en-GB',{timeZone:'Europe/Copenhagen',dateStyle:'short',timeStyle:'short'});
      for(const row of data.rows){const tr=document.createElement('tr');for(const value of [dates.format(new Date(row.time_utc)),format(row.price_dkk_mwh)]){const td=document.createElement('td');td.textContent=value;tr.append(td);}el('latestRows').append(tr);}
      el('latestStatus').textContent=`${data.rows.length} published intervals. Retrieved ${dates.format(new Date(data.fetched_at))} Copenhagen time.`;
    }catch(error){el('latestStatus').textContent=error.message;}finally{el('latestLoad').disabled=false;}
  });
  window.addEventListener('pagehide',pauseReplay);
  el('editSignal').addEventListener('click',()=>{
    pauseReplay();el('strategyEditor').open=true;el('strategyEditor').scrollIntoView({behavior:'auto',block:'start'});el('labLower').focus({preventScroll:true});
  });
  el('signalCanvas').addEventListener('mousemove',event=>{
    const bounds=el('signalCanvas').getBoundingClientRect();drawSignal((event.clientX-bounds.left-65)/(bounds.width-81));
  });
  el('signalCanvas').addEventListener('mouseleave',()=>drawSignal());
  // The inline editor behaves like a one-line code cell: Enter runs it, Shift+Enter adds a line.
  el('labFormula').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();el('labForm').requestSubmit();}});
  window.addEventListener('resize',draw); buildFactors(); buildLibrary(); updateFields();
  el('labForm').requestSubmit();
})();
