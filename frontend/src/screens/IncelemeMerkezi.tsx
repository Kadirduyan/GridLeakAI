/** İnceleme Merkezi Ekranı — Risk sırasına göre abone seçimi, tüketim grafiği, TreeSHAP açıklaması, veri kalitesi ve analist notu yönetimi. */
import {useState} from 'react';
import {quality,type Subscriber} from '../data';
import AboneTablosu from '../shared-components/AboneTablosu';
import SkorGostergesi from '../shared-components/SkorGostergesi';
import VeriKalitesi from '../shared-components/VeriKalitesi';
import OncelikSirasi from '../shared-components/OncelikSirasi';
import TuketimGrafigi from '../shared-components/TuketimGrafigi';
import ModelAciklamasi from '../shared-components/ModelAciklamasi';
function AnalystNote({subscriber,note,onSave,onPlan}:{subscriber:Subscriber;note:string;onSave:(s:string)=>Promise<boolean>;onPlan:()=>void}){
  const [draft,setDraft]=useState(note),[saved,setSaved]=useState(false);
  return <section className="gl-panel"><h3 className="gl-paneltitle">Analist değerlendirmesi</h3><div className="gl-notes"><label htmlFor="analyst-note" className="gl-small">Kontrol notu</label><textarea id="analyst-note" maxLength={4000} placeholder="Müşteri açıklaması, sayaç kontrolü ve takip adımları…" value={draft} onChange={e=>{setDraft(e.target.value);setSaved(false)}}/><div className="gl-actions"><button className="gl-action" onClick={async()=>{setSaved(await onSave(draft))}}>Notu kaydet</button><button className="gl-action" onClick={onPlan}>Denetim planını aç</button></div></div><div className="gl-status" aria-live="polite">{draft!==note?'Kaydedilmemiş değişiklik var. Abone değiştirmeden önce kaydedin.':saved?`${subscriber.id} için not kaydedildi.`:note?'Kaydedilmiş analist notu yüklendi.':'Henüz kaydedilmiş analist notu yok.'}</div></section>;
}
export default function IncelemeMerkezi({subscribers,selected,onSelect,capacity,notes,onSave,onPlan}:{subscribers:Subscriber[];selected:Subscriber;onSelect:(a:Subscriber)=>void;capacity:number;notes:Record<string,string>;onSave:(id:string,s:string)=>Promise<boolean>;onPlan:()=>void}){
  const [search,setSearch]=useState('');
  return <section className="gl-workspace"><AboneTablosu subscribers={subscribers} selected={selected.id} search={search} onSearch={setSearch} onSelect={onSelect}/><div className="gl-main"><div className="gl-summaryhead"><h2>Abone karar özeti · #{selected.id} · {selected.partition==='training'?'Eğitim abonesi':'Test abonesi'}</h2><span className={`gl-tag ${quality(selected)<90?'amber':''}`}>{quality(selected)>=90?'Model destekli analist incelemesi':'Öneri: önce veri kontrolü'}</span></div><div className="gl-cards"><SkorGostergesi score={selected.score}/><VeriKalitesi subscriber={selected}/><OncelikSirasi subscriber={selected} capacity={capacity} total={subscribers.filter(a=>a.score!==null).length}/></div><TuketimGrafigi key={selected.id} subscriber={selected}/><div className="gl-lower"><ModelAciklamasi id={selected.id}/><AnalystNote key={selected.id} subscriber={selected} note={notes[selected.id]||''} onSave={s=>onSave(selected.id,s)} onPlan={onPlan}/></div></div></section>;
}
