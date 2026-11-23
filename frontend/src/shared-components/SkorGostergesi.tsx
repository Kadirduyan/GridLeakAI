import {useId} from 'react';
import {nf} from '../data';
const threshold=0.29006222348325456;
export default function SkorGostergesi({score}:{score:number|null}){
 const id=useId(),valid=score!==null&&Number.isFinite(score),value=valid?Math.max(0,Math.min(1,score)):0;
 const tier=!valid?'Skorlanamadı':value>=0.40294978717998475?'Kritik':value>=threshold?'Birincil':value>=0.19211300698790673?'Geniş':'Düşük';
 const tone=!valid?'muted':tier==='Kritik'?'critical':tier==='Birincil'?'primary':tier==='Geniş'?'wide':'low';
 const angle=Math.PI*(1-threshold),mx=(radius:number)=>140+radius*Math.cos(angle),my=(radius:number)=>140-radius*Math.sin(angle);
 return <section className="gl-panel gl-metric score-card"><h3>01 · MODEL SKORU</h3>
 <svg className="gl-gauge" viewBox="0 0 280 180" role="img" aria-label={valid?`Kalibre risk puanı ${nf(value*100)} / 100. Karar eşiği ${nf(threshold*100)}.`:'Abone skorlanamadı'}>
 <defs><linearGradient id={id} x1="0" x2="1"><stop stopColor="#42e5cf"/><stop offset=".45" stopColor="#ffc465"/><stop offset="1" stopColor="#ff567a"/></linearGradient></defs>
 <path d="M 30 140 A 110 110 0 0 1 250 140" fill="none" stroke="#27374a" strokeWidth="20"/>
 {valid&&<><path d="M 30 140 A 110 110 0 0 1 250 140" pathLength="100" strokeDasharray={`${value*100} 100`} fill="none" stroke={`url(#${id})`} strokeWidth="20"/>
 <line x1={mx(96)} y1={my(96)} x2={mx(125)} y2={my(125)} stroke="#fff3c7" strokeWidth="3"/><text x="140" y="116" textAnchor="middle" className="gauge-value">{nf(value*100)}</text><text x="140" y="139" textAnchor="middle" className="gauge-caption">/ 100 puan</text></>}
 {!valid&&<text x="140" y="118" textAnchor="middle" fill="#c6d3e4" fontSize="20">Skorlanamadı</text>}
 <text x="30" y="168" className="gauge-caption">0</text><text x="250" y="168" textAnchor="end" className="gauge-caption">100</text></svg>
 <p className="gl-small">Kilitli karar eşiği: <strong>{nf(threshold*100)}</strong>{valid&&` · ${nf(Math.abs(value-threshold)*100)} puan ${value>=threshold?'üzerinde':'altında'}`}</p>
 <span className={`risk-badge ${tone}`}>{tier}{valid?' risk':''}</span>
 <p className="gl-explain">{valid?'Kalibre model olasılığının 0–100 puan gösterimi. Kesin kaçak tespiti değildir.':'Bu abone için kullanılabilir model skoru bulunmuyor. Veri kayıtlarını kontrol edin.'}</p></section>;
}