/** GridLeak AI — Ana uygulama kabuğu; 3 ekran arası geçişi, abone yüklemeyi ve backend durum senkronizasyonunu yönetir. */
import {useEffect,useRef,useState} from 'react';
import Menu from './shared-components/Menu';
import IncelemeMerkezi from './screens/IncelemeMerkezi';
import DenetimPlani from './screens/DenetimPlani';
import SahaSonuclari from './screens/SahaSonuclari';

import {downloadCsv,pages,quality,type Page,type SavedState,type Subscriber} from './data';
import {request,loadState,loadSubscribers,loadConsumption} from './api';
function pageFromHash():Page {const id=location.hash.replace('#/','');if(pages.some(p=>p.id===id))return id as Page;history.replaceState(null,'',location.pathname+location.search+'#/review');return 'review'}
export default function App(){
 const [page,setPage]=useState<Page>(pageFromHash),[subscribers,setSubscribers]=useState<Subscriber[]>([]),[selectedId,setSelectedId]=useState('');
 const [state,setState]=useState<SavedState>({plan:[],notes:{},results:{},capacity:20}),[message,setMessage]=useState(''),[error,setError]=useState(''),[loading,setLoading]=useState(true),[busy,setBusy]=useState(false);
 const locked=useRef(false),timer=useRef<ReturnType<typeof setTimeout>|null>(null);
 function notify(text:string){setMessage(text);if(timer.current)clearTimeout(timer.current);timer.current=setTimeout(()=>setMessage(''),6500)}
 async function load(){setLoading(true);setError('');try{const [list,s]=await Promise.all([loadSubscribers(),loadState()]);setSubscribers(list);setState(s);setSelectedId(id=>list.some(a=>a.id===id)?id:list[0]?.id||'')}catch(e){setError(e instanceof Error?e.message:'Veriler yüklenemedi.')}finally{setLoading(false)}}
 useEffect(()=>{void load();const fn=()=>setPage(pageFromHash());window.addEventListener('hashchange',fn);return()=>{window.removeEventListener('hashchange',fn);if(timer.current)clearTimeout(timer.current)}},[]);
 async function mutate(path:string,method:string,body:unknown,success:string):Promise<boolean>{
  if(locked.current)return false;locked.current=true;setBusy(true);
  try{await request(path,method,body);const s=await loadState();setState(s);notify(success);return true}catch(e){notify(e instanceof Error?e.message:'İşlem tamamlanamadı.');return false}finally{locked.current=false;setBusy(false)}
 }
 function go(p:Page){setPage(p);location.hash='/'+p}
 function review(a:Subscriber){setSelectedId(a.id);go('review')}
 useEffect(()=>{let cancelled=false;const a=subscribers.find(a=>a.id===selectedId);if(a&&!a.points.length)void loadConsumption(a).then(full=>{if(!cancelled)setSubscribers(items=>items.map(x=>x.id===full.id?full:x))}).catch(e=>{if(!cancelled)setError(String(e))});return()=>{cancelled=true}},[selectedId,subscribers]);
 const selected=subscribers.find(a=>a.id===selectedId);
 function togglePlan(){if(!selected)return;const exists=state.plan.includes(selected.id);void mutate('/plan/'+encodeURIComponent(selected.id),exists?'DELETE':'PUT',undefined,exists?'Abone plandan çıkarıldı; kayıtlar korundu.':'Abone denetim planına kaydedildi.')}
 function exportList(){const list=page==='plan'?subscribers.filter(a=>state.plan.includes(a.id)):page==='results'?subscribers.filter(a=>state.plan.includes(a.id)||state.results[a.id]):subscribers;if(!list.length){notify('Bu listede indirilecek kayıt yok.');return}downloadCsv(`gridleak-${page}-ornek.csv`,[['veri_turu','abone','model_skoru','kayit_dolulugu_yuzde','portfoy_sirasi','planda','saha_sonucu','analist_notu','saha_notu'],...list.map(a=>['SGCC',a.id,a.score??'',quality(a),a.rank??'',state.plan.includes(a.id)?'Evet':'Hayır',state.results[a.id]?.status||'',state.notes[a.id]||'',state.results[a.id]?.note||''])]);notify('Sunucudan alınan kayıtlar CSV olarak indirildi.')}
 const meta=pages.find(p=>p.id===page)!;
 return <div id="gl-console"><a href="#main-content" className="skip-link">İçeriğe geç</a><header className="gl-top"><div className="gl-brand"><svg viewBox="0 0 32 32" width="25" height="28" fill="none" stroke="currentColor" strokeWidth="1.7" aria-hidden="true"><path d="M5 9h22M8 5v8m8-8v8m8-8v8M5 9l11 11L27 9M16 20v11"/></svg><span>GridLeak<small>ENERJİ DENETİM PLATFORMU</small></span></div><span className="gl-demo">YEREL UYGULAMA · SGCC ABONE VERİSİ</span></header><Menu page={page} onChange={go}/><main className="gl-body" id="main-content"><div className="gl-heading"><div><h1>{meta.title}</h1><p className="gl-sub">{meta.subtitle}</p></div><div className="gl-actions"><button className="gl-action" disabled={loading||busy} onClick={()=>void load()}>Verileri yenile</button>< ><button className="gl-action" disabled={loading||!!error} onClick={exportList}>Listeyi CSV indir</button>{page==='review'&&selected&&<button className="gl-action primary" disabled={busy||loading||!!error} onClick={togglePlan}>{state.plan.includes(selected.id)?'Plandan çıkar':'Denetime ekle'}</button>}</></div></div>
 {loading?<section className="gl-panel gl-empty" role="status">Sunucudan veriler alınıyor…</section>:error?<section className="gl-panel gl-empty" role="alert"><h2>Bağlantı kurulamadı</h2><p>{error}</p><p>Backend'i başlatın ve tekrar deneyin. Eski örnek veriler sessizce gösterilmez.</p><button className="gl-action" onClick={()=>void load()}>Tekrar dene</button></section>:<fieldset disabled={busy} style={{border:0,padding:0,margin:0,minWidth:0}}>
 {page==='review'&&(selected?.points.length?<IncelemeMerkezi subscribers={subscribers} selected={selected} onSelect={a=>setSelectedId(a.id)} capacity={state.capacity} notes={state.notes} onSave={(id,note)=>mutate('/subscribers/'+encodeURIComponent(id)+'/note','PUT',{note},'Analist notu sunucuya kaydedildi.')} onPlan={()=>go('plan')}/>:<div className="gl-empty">Abone kaydı bulunamadı.</div>)}
 {page==='plan'&&<DenetimPlani subscribers={subscribers} state={state} onCapacity={capacity=>mutate('/plan/capacity','PUT',{capacity},'Kapasite kaydedildi.')} onRemove={id=>void mutate('/plan/'+encodeURIComponent(id),'DELETE',undefined,'Plandan çıkarıldı; sonuçlar korundu.')} onReview={review} onResults={()=>go('results')}/>}
 {page==='results'&&<SahaSonuclari subscribers={subscribers} state={state} onSave={(id,r)=>mutate('/results/'+encodeURIComponent(id),'PUT',{status:r.status,note:r.note},'Saha sonucu sunucuya kaydedildi.')} onPlan={()=>go('plan')}/>}
</fieldset>}
 <footer className="gl-foot"><span>SGCC kayıtları · Kayıtlar SQLite veritabanında</span><span>Kalibre risk puanı · Nihai değerlendirme saha incelemesiyle yapılır.</span></footer></main><div className={`toast ${message?'visible':''}`} role="status" aria-live="polite">{message}</div></div>
}
