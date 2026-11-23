import {quality,type Subscriber} from '../data';
export default function VeriKalitesi({subscriber:a}:{subscriber:Subscriber}){
  const q=quality(a),good=q>=90;
  return <section className="gl-panel gl-metric quality"><h3>02 · VERİ KALİTESİ</h3><svg className="gl-chart" viewBox="0 0 220 135" role="img" aria-label={`Kayıt doluluğu yüzde ${q}; ${365-a.missing.length} gün`}>
    <circle cx="110" cy="67" r="49" fill="none" stroke="#233044" strokeWidth="10"/><circle cx="110" cy="67" r="49" fill="none" stroke={good?'#11b991':'#e3b76a'} strokeWidth="10" pathLength="100" strokeDasharray={`${(365-a.missing.length)/365*100} 100`} transform="rotate(-90 110 67)"/>
    <text x="110" y="71" textAnchor="middle" fill="#f3f6fc" fontSize="32" fontWeight="650">%{q}</text><text x="110" y="93" textAnchor="middle" fill="#9edeca" fontSize="11">{365-a.missing.length} / 365 gün</text></svg>
    <span className="gl-small">Kayıt sürekliliği · kırmızı: eksik</span>
    <div className="gl-strip" role="img" aria-label={`Yıllık kayıt sürekliliği: ${a.missing.length} gün eksik. Her çizgi beş günü temsil eder.`}>{Array.from({length:73},(_,i)=>{const count=a.missing.filter(d=>Math.floor(d/5)===i).length;return <i key={i} className={count?'gl-missing':''} title={`${i*5+1}–${Math.min(365,i*5+5)}. gün: ${count} eksik kayıt`}/>})}</div>
    <p className="gl-explain">{good?'Analiz için kayıt tamlığı yeterli.':'Kayıt tamlığı düşük; saha planından önce veri kontrolü önerilir.'} {a.missing.length} günlük ölçüm eksik.</p></section>;
}
