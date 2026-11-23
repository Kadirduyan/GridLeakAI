import {nf,quality,type Subscriber} from '../data';
export default function OncelikSirasi({subscriber:a,capacity,total}:{subscriber:Subscriber;capacity:number;total:number}){
 const ranked=a.rank!==null&&total>0,position=ranked?Math.max(0,Math.min(100,(a.rank!-1)/Math.max(1,total-1)*100)):0,inside=ranked&&a.rank!<=capacity;
 return <section className="gl-panel gl-metric priority-card"><h3>03 · DENETİM ÖNCELİĞİ</h3><div className="gl-ranknum">{ranked?'#'+nf(a.rank!):'—'}</div>
 <p className="gl-small">{ranked?`${nf(total)} abone içinde ${a.rank!/total*100<.1?'ilk %0,1':'ilk %'+nf(a.rank!/total*100)}`:'Bu abone için sıralama yok.'}</p>
 <div className="gl-ranktrack" role="img" aria-label={ranked?`${total} abone içinde sıra ${a.rank}`:'Sıra yok'}>{ranked&&<i style={{left:`${position}%`}}/>}</div><div className="gl-rankaxis"><span>#1</span><span>#{nf(total)}</span></div>
 <div className="gl-mini"><span className={`risk-badge ${inside?'low':'wide'}`}>{!ranked?'Sıralama dışı':inside?'Kapasite içinde':'Kapasite dışında'}</span><p>Hedef kapasite: ilk <strong>{nf(capacity)}</strong> abone</p><p>{quality(a)>=90?'Veri tamlığı uygun.':'Önce eksik kayıt kontrolü gerekli.'}</p></div><p className="gl-explain">Portföy sırası risk skoruna göredir. Kapasite içinde olmak plana otomatik eklenmek anlamına gelmez.</p></section>;
}