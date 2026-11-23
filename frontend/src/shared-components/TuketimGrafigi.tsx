import {useState} from 'react';
import {consumption,nf,type Subscriber} from '../data';
export default function TuketimGrafigi({subscriber:a}:{subscriber:Subscriber}){
  const [period,setPeriod]=useState(90),[hover,setHover]=useState<number|null>(null);
  const vals=consumption(a),start=a.points.length-period,arr=vals.slice(start);
  const w=900,h=305,l=64,r=28,t=28,b=36,max=26;
  const x=(i:number)=>l+i*(w-l-r)/(arr.length-1),y=(v:number)=>t+(max-v)*(h-t-b)/max;
  function path(vs:(number|null)[]){let active=false;return vs.map((v,i)=>{if(v===null){active=false;return ''}const s=(active?'L':'M')+x(i).toFixed(1)+' '+y(v).toFixed(1);active=true;return s}).join(' ')}
  const avg=arr.map((v,i)=>{if(v===null)return null;const slice=vals.slice(Math.max(0,start+i-6),start+i+1).filter((n):n is number=>n!==null);return slice.reduce((s,n)=>s+n,0)/slice.length});
  const mark=-1;
  const fmt=(i:number)=>new Date(a.points[start+i].date+'T00:00:00Z').toLocaleDateString('tr-TR',{day:'numeric',month:'short',timeZone:'UTC'});
  const hv=hover===null?null:arr[hover];
  return <section className="gl-panel"><div className="gl-charthead"><div><h3>Günlük elektrik tüketimi</h3><p className="gl-sub">Örnek tüketim ve 7 günlük hareketli ortalama · kWh/gün</p></div><div className="gl-periods" aria-label="Grafik zaman aralığı">{[[30,'30 gün'],[90,'90 gün'],[365,'1 yıl']].map(([value,label])=><button key={value} aria-pressed={period===value} onClick={()=>{setPeriod(Number(value));setHover(null)}}>{label}</button>)}</div></div>
    <div className="gl-consumption"><svg viewBox={`0 0 ${w} ${h}`} role="img" tabIndex={0} aria-label={`${period} günlük örnek tüketim grafiği. Günler arasında gezinmek için sağ ve sol okları kullanın.`}
      onPointerMove={e=>{const box=e.currentTarget.getBoundingClientRect();const px=(e.clientX-box.left)/box.width*w;setHover(Math.max(0,Math.min(period-1,Math.round((px-l)/(w-l-r)*(period-1)))))}}
      onPointerLeave={()=>setHover(null)} onBlur={()=>setHover(null)}
      onKeyDown={e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();setHover(i=>Math.max(0,Math.min(period-1,(i??0)+(e.key==='ArrowRight'?1:-1))))}if(e.key==='Escape')setHover(null)}}>
      {[0,8,16,24].map(v=><g key={v}><line x1={l} y1={y(v)} x2={w-r} y2={y(v)} stroke="#273445"/><text x={l-10} y={y(v)+4} textAnchor="end" fill="#a6b5c9" fontSize="13">{v}</text></g>)}
      {mark>=0&&<g><line x1={x(mark)} y1={t} x2={x(mark)} y2={h-b} stroke="#d5ae68" strokeDasharray="3 4"/><text x={Math.min(x(mark)+7,w-205)} y="20" fill="#dfba7d" fontSize="13">Tüketim düzeyi değişimi</text></g>}
      <path d={path(arr)} fill="none" stroke="#50d2bd" strokeWidth="2"/><path d={path(avg)} fill="none" stroke="#b6c2d5" strokeWidth="2.2" strokeDasharray="6 4"/>
      {[0,Math.floor((period-1)/2),period-1].map(i=><text key={i} x={x(i)} y={h-9} textAnchor={i===0?'start':i===period-1?'end':'middle'} fill="#a6b5c9" fontSize="13">{fmt(i)}</text>)}
      {hover!==null&&<g pointerEvents="none"><line x1={x(hover)} x2={x(hover)} y1={t} y2={h-b} stroke="#91a3b9" strokeDasharray="3 4"/>{hv!==null&&<circle cx={x(hover)} cy={y(hv)} r="4" fill="#50d2bd" stroke="#101923" strokeWidth="2"/>}<g transform={`translate(${Math.min(Math.max(x(hover)-82,l),w-192)},${t+8})`}><rect width="175" height="65" rx="5" fill="#1b2837" stroke="#415267"/><text x="12" y="22" fill="#eef3fa" fontSize="13">{fmt(hover)} · {hv===null?'Kayıt eksik':`${nf(hv)} kWh`}</text><text x="12" y="47" fill="#adbed1" fontSize="12">7 gün ort.: {avg[hover]===null?'—':nf(avg[hover]!)+' kWh'}</text></g></g>}
    </svg></div><div className="gl-legend"><span><i/>Günlük tüketim</span><span><i className="avg"/>7 günlük ortalama</span><span>Boşluk = eksik kayıt · 0 = ölçülmüş sıfır tüketim</span></div>
    <div className="gl-status">Sentetik tüketim verisi API’den alınır. Eksik kayıtlar sıfıra çevrilmez. Model sonucu veya gerçek kaçak bulgusu değildir.</div></section>;
}
