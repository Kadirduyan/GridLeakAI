import {useId} from 'react';
import {nf,type Subscriber} from '../data';
export default function VeriKalitesi({subscriber:a}:{subscriber:Subscriber}){
 const id=useId(),total=a.points.length,missing=a.points.filter(p=>p.consumption_kwh===null).length,q=total?(total-missing)/total*100:0;
 return <section className="gl-panel gl-metric quality"><h3>02 · VERİ KALİTESİ</h3><svg className="quality-ring" viewBox="0 0 240 180" role="img" aria-label={`Kayıt doluluğu yüzde ${nf(q)}; ${total-missing} / ${total} gün`}>
 <defs><linearGradient id={id} x2="1" y2="1"><stop stopColor="#64f0dc"/><stop offset="1" stopColor="#11b9b0"/></linearGradient></defs>
 <circle cx="120" cy="86" r="65" fill="none" stroke="#25364a" strokeWidth="12"/><circle className="ring-glow" cx="120" cy="86" r="65" fill="none" stroke={`url(#${id})`} strokeWidth="12" pathLength="100" strokeDasharray={`${q} 100`} transform="rotate(-90 120 86)" strokeLinecap="round"/>
 <text x="120" y="88" textAnchor="middle" fill="#f4fbff" fontSize="29" fontWeight="700">%{nf(q)}</text><text x="120" y="111" textAnchor="middle" fill="#b3d9d8" fontSize="12">{total-missing} / {total} gün</text></svg>
 <span className="gl-small">Kayıt sürekliliği · her çubuk 5 gün</span><div className="gl-strip" role="img" aria-label={`${missing} eksik gün, ${total-missing} kayıtlı gün`}>{Array.from({length:Math.ceil(total/5)},(_,i)=>{const chunk=a.points.slice(i*5,(i+1)*5),missCount=chunk.filter(p=>p.consumption_kwh===null).length;return <i key={i} className={missCount?'gl-missing':''} title={`${chunk[0]?.date} – ${chunk[chunk.length-1]?.date}: ${missCount?`${missCount} eksik gün`:'tam kayıt'}`}/>})}</div>
 <div className="strip-legend"><span>● Kayıt var</span><span>● Eksik</span></div><p className="gl-explain">{q>=90?'Kayıt tamlığı yeterli.':'Önce veri kontrolü önerilir.'} {missing} günlük kayıt eksik. Eksik kayıt, sıfır tüketim değildir.</p></section>;
}