import {useEffect,useState} from 'react';
import {request} from '../api';
import {nf} from '../data';
type E={message:string;explanation:null|{base_value:number;features:{feature:string;contribution:number}[]}};
function label(f:string){
 if(/post_missing.*drop/.test(f))return 'Eksik kayıt sonrası tüketim düşüşü';
 if(/yoy|2016.*2014|2016.*2015/.test(f))return 'Yıllık tüketim değişimi';
 if(/longest.*zero|zero.*run|sifir_serisi/.test(f))return 'Ardışık sıfır tüketim';
 if(/zero|sifir/.test(f))return 'Sıfır tüketim düzeni';
 if(/missing|miss|gap/.test(f))return 'Eksik kayıt sıklığı ve süresi';
 if(/svd/.test(f))return 'Tüketim örüntüsü bileşeni';
 if(/ctx_|peer_z_/.test(f))return 'Benzer abonelerden ve şebekeden sapma';
 if(/drop|decline|dus|oct.*vs/.test(f))return 'Dönemler arası tüketim düşüşü';
 if(/std|cv|mad|standart/.test(f))return 'Tüketim değişkenliği';
 if(/wk|week|dow|hafta/.test(f))return 'Haftalık tüketim düzeni';
 if(/peak|max|en_yuksek/.test(f))return 'Yüksek tüketim ve ani zirveler';
 if(/step|jump|cusum|slope/.test(f))return 'Tüketim düzeyindeki değişim';
 if(/flat/.test(f))return 'Uzun süre sabit tüketim';
 if(/mean|median|ortalama|ortanca/.test(f))return 'Tipik tüketim düzeyi';
 if(/corr/.test(f))return 'Tüketimin zaman içindeki benzerliği';
 return 'Dönemsel tüketim örüntüsü';
}
export default function ModelAciklamasi({id}:{id:string}){
 const [data,setData]=useState<E|null>(null);
 useEffect(()=>{let active=true;setData(null);void request<E>(`/subscribers/${encodeURIComponent(id)}/explanation`).then(d=>{if(active)setData(d)}).catch(e=>{if(active)setData({message:String(e),explanation:null})});return()=>{active=false}},[id]);
 const all=(data?.explanation?.features||[]).filter(f=>Number.isFinite(f.contribution)),top=[...all].sort((a,b)=>Math.abs(b.contribution)-Math.abs(a.contribution)).slice(0,6),sum=top.reduce((s,f)=>s+Math.abs(f.contribution),0);
 return <section className="gl-panel"><h3 className="gl-paneltitle">Model kararını etkileyen etkenler</h3><div className="gl-feature">{data?.explanation?<><p className="gl-small">En güçlü 6 etken · LightGBM bileşeninin açıklaması</p>{top.map((f,i)=>{const share=sum?Math.abs(f.contribution)/sum*100:0;return <div className="gl-featureline" key={f.feature}><span title={f.feature}>{label(f.feature)}{top.slice(0,i).some(x=>label(x.feature)===label(f.feature))&&<small> · farklı ölçüm</small>}</span><div className="gl-track"><i className={f.contribution<0?'down':''} style={{width:`${share}%`}}/></div><strong className={f.contribution<0?'text-teal':'text-pink'}>{f.contribution<0?'-':'+'}%{nf(share)}</strong></div>})}
 <p className="gl-small">Oranlar, gösterilen altı etkenin toplam mutlak katkısı içindeki paydır; risk olasılığındaki yüzde puan değişimi değildir.</p><details key={id} className="shap-details"><summary>Teknik detay (SHAP)</summary><p>Referans değer: {nf(data.explanation.base_value)} · ham model skoru (log-odds)</p>{all.map(f=><div key={f.feature} className="raw-contribution"><code>{f.feature}</code><span>{f.contribution.toFixed(4)}</span></div>)}</details></>:<p role="status">{data?.message||'Etkenler yükleniyor…'}</p>}</div><p className="gl-status">Kırmızı: riski artıran etki · Turkuaz: riski azaltan etki · Kesin kaçak kanıtı değildir.</p></section>;
}