export async function buildFaDeck(presentation, session, content, data, design, widths) {
  const C = {bg:'#0b1220',fg:'#edf3fc',muted:'#a9bbd2',green:'#43ddb2',blue:'#72b7ff',red:'#ff8795',gold:'#f3c879'};
  presentation.slides.keep([]);
  presentation.theme.defaultFont = 'Tahoma';
  function atom(sl, value, x,y,w,h,size,color,bold,align='right') {
    const shape=sl.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
    shape.text=value;
    shape.text.style={fontSize:size,typeface:'Tahoma',color,bold,alignment:align,verticalAlignment:'middle',wrap:'none',autoFit:'none',insets:0};
    return shape;
  }
  function text(sl,value,x,y,w,h,size=28,color=C.fg,bold=false,align='right') {
    if (!/[\u0600-\u06ff]/.test(value)) return atom(sl,value,x,y,w,h,size,color,bold,align);
    // Each logical Persian word stays editable. Separate word boxes enforce the reading
    // order in the current renderer, which does not expose paragraph RTL direction.
    const placements=[]; let cursor=x+w, line=0;
    for(const word of value.split(/\s+/)) {
      const measured=widths[word]?.[bold?1:0] ?? word.length*58;
      const ww=Math.ceil(measured*size/100+5);
      if(cursor-ww<x && cursor<x+w) {line++;cursor=x+w;}
      placements.push({word,left:cursor-ww,line,width:ww});cursor-=ww+size*.23;
    }
    const lineH=size*1.45, blockH=(line+1)*lineH;
    if(blockH>h+3) throw new Error('Persian text overflow: '+value+' needs '+blockH+'px, has '+h);
    const yy=y+(h-blockH)/2;
    placements.forEach(p=>atom(sl,p.word,p.left,yy+p.line*lineH,p.width,lineH,size,color,bold));
  }
  function base(i) {
    const sl=presentation.slides.add({width:1280,height:720});sl.background.fill=C.bg;
    if(i!==0) text(sl,content.slides[i].title,56,40,1168,80,42,C.fg,true);
    atom(sl,String(i+1).padStart(2,'0'),56,672,55,25,17,C.muted,false,'left');
    sl.speakerNotes.text=content.slides[i].script+'\n\nزمان پیشنهادی: '+content.slides[i].duration+' ثانیه\nمنابع: '+content.slides[i].source;
    return sl;
  }
  function table(sl, values, y, columnWidths, h, font=24) {
    const tb=sl.tables.add({rows:values.length,columns:values[0].length,left:80,top:y,width:1120,height:h,columnWidths,values});
    tb.borders.assign({fill:C.bg,width:0,style:'solid'});
    for(let r=0;r<values.length;r++) {
      tb.rows[r].height=h/values.length;
      for(let c=0;c<values[r].length;c++) {
        const cell=tb.getCell(r,c);cell.fill=C.bg;
        cell.text.style={fontSize:font,typeface:'Tahoma',alignment:'right',color:r===0?C.green:C.fg,bold:r===0,insets:2};
      }
    }
    return tb;
  }
  function chart(sl,series,pos,min,max,legend=false) {
    return sl.charts.add('line',{position:pos,categories:data.chart.categories,series,
      hasLegend:legend,legend:{position:'bottom',overlay:false,textStyle:{fontSize:17,fill:C.muted}},
      xAxis:{visible:false},yAxis:{visible:true,min,max,numberFormatCode:'0%',textStyle:{fontSize:19,fill:C.muted},majorGridlines:{fill:'#293a50',width:1,style:'solid'}},
      chartFill:C.bg,plotAreaFill:C.bg,chartLine:{fill:'none',width:0},plotAreaLine:{fill:'none',width:0},
      lineOptions:{smooth:false},dataLabels:{showValue:false}});
  }
  let s=base(0);
  text(s,design.cover[0],90,170,1100,110,65,C.fg,true);
  text(s,design.cover[1],90,300,1100,65,32,C.green);
  text(s,design.cover[2],90,390,1100,60,28,C.muted);
  text(s,design.cover[3],90,530,1100,48,25,C.fg);
  text(s,design.cover[4],90,590,1100,42,23,C.muted);
  s=base(1);
  design.steps.forEach((r,k)=>{const y=180+k*110;atom(s,r[0],1145,y,60,70,46,C.green,true);text(s,r[1],650,y,470,45,31,C.fg,true);text(s,r[2],80,y+45,1040,46,25,C.muted);});
  s=base(2);
  design.risk.forEach((v,k)=>text(s,v,80,[170,245,325,410,485,570][k],1120,k===0?66:55,[36,25,27,29,29,27][k],k===0?C.green:k===4?C.red:k===1||k===5?C.muted:C.fg,k===0));
  s=base(3);
  text(s,design.backtest[0],650,135,550,70,48,C.green,true);
  text(s,design.backtest[1],80,145,550,55,29,C.fg,true);
  text(s,design.backtest[2],630,208,570,45,25,C.muted);
  text(s,design.backtest[3],80,208,540,45,25,C.muted);
  chart(s,[{name:'S1 (TMN)',values:data.chart.toman,line:{fill:C.green,width:3}},
    {name:'Hold USDT (TMN)',values:data.chart.usdt_hold_toman,line:{fill:C.gold,width:2}}],{left:80,top:270,width:1120,height:295},-.02,.24,true);
  text(s,design.backtest[4]+'، '+design.backtest[5],80,570,1120,40,22,C.muted);
  text(s,design.backtest[6],80,615,1120,34,19,C.muted);
  text(s,design.backtest[7],125,657,1075,35,18,C.muted);
  s=base(4);
  text(s,design.currency[0].split(' ')[0],660,140,540,70,48,C.blue,true);
  text(s,design.currency[0].split(' ').slice(1).join(' '),660,212,540,45,27,C.blue);
  text(s,design.currency[1].split(' ')[0],80,140,550,70,48,C.red,true);
  text(s,design.currency[1].split(' ').slice(1).join(' '),80,212,550,45,27,C.red);
  chart(s,[{name:'USDT return',values:data.chart.usdt,line:{fill:C.blue,width:2.5}}],{left:665,top:275,width:535,height:265},-.01,.13);
  chart(s,[{name:'TMN drawdown',values:data.chart.drawdown,line:{fill:C.red,width:2.5}}],{left:80,top:275,width:535,height:265},-.045,0);
  text(s,design.currency[2],80,565,1120,48,27,C.gold);
  text(s,design.currency[3],130,638,1070,55,18,C.muted);
  s=base(5);
  text(s,design.shadow[0],80,130,1120,50,28,C.fg,true);
  text(s,design.shadow[1],80,185,1120,40,25,C.red,true);
  text(s,design.shadow[2],80,228,1120,35,21,C.muted);
  const faNum=x=>Math.round(x).toLocaleString('en-US');
  const closed=[['خروج','بازده','تومان','ورود','جفت'],...data.closed.map(r=>[
    r.exit_reason==='agent_exit'?'GPT':'SL', (r.ret_pct*100).toFixed(2)+'%',faNum(r.pnl_tmn),r.entry_tehran.split(' ').slice(-1)[0],r.symbol.replace('TMN','/TMN')])];
  table(s,closed,270,[150,175,245,260,290],340,23);
  text(s,design.shadow[3],80,620,1120,34,20,C.muted);
  text(s,design.shadow[4],130,660,1070,32,18,C.muted);
  s=base(6);
  text(s,design.open[0],80,155,1120,55,30,C.fg,true);
  const unfinished=[['وضعیت','مقدار','ورود','جفت'],...data.open.map(r=>['ناتمام',String(r.qty),r.entry_tehran.split(' ').slice(-1)[0],r.symbol.replace('TMN','/TMN')])];
  table(s,unfinished,240,[260,260,290,310],190,25);
  text(s,design.open[1],80,452,1120,45,24,C.muted);
  text(s,design.open[2],80,535,1120,52,31,C.green,true);
  text(s,design.open[3],130,611,1070,55,23,C.muted);
  s=base(7);
  text(s,design.example[0],80,155,1120,50,30,C.fg,true);
  text(s,design.example[1],80,220,1120,55,35,C.green,true);
  text(s,design.example[2],80,295,1120,50,30,C.fg);
  text(s,design.example[3],80,365,1120,45,27,C.muted);
  text(s,design.example[4],80,443,1120,55,28,C.fg);
  text(s,design.example[5],80,525,1120,55,32,C.red,true);
  text(s,design.example[6],130,626,1070,55,22,C.muted);
  s=base(8);
  text(s,design.finish[0],80,155,1120,58,36,C.green,true);
  text(s,design.finish[1],80,230,1120,50,28,C.fg);
  text(s,design.finish[2],80,310,1120,50,28,C.fg);
  text(s,design.finish[3],80,395,1120,58,28,C.muted);
  text(s,design.finish[4],80,485,1120,55,29,C.gold);
  text(s,design.finish[5],130,585,1070,60,27,C.fg);
  session.setActiveSlide(presentation.slides.getItem(0));
  return {slideCount:presentation.slides.items.length,narrationSeconds:content.slides.reduce((n,s)=>n+s.duration,0)};
}
