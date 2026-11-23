// Backend ile bütün HTTP iletişimi burada; arayüz doğrudan veritabanına erişmez.
import type {SavedState, Subscriber, Status} from './data';

export async function request<T>(path:string, method='GET', body?:unknown):Promise<T> {
  const controller=new AbortController();
  const timer=setTimeout(()=>controller.abort(),15000);
  try {
    const response=await fetch('/api'+path,{
      method, signal:controller.signal,
      headers:body===undefined?undefined:{'Content-Type':'application/json'},
      body:body===undefined?undefined:JSON.stringify(body)
    });
    const data=await response.json().catch(()=>null);
    if(!response.ok) throw new Error(typeof data?.detail==='string'?data.detail:`İstek tamamlanamadı (${response.status}). Alanları kontrol edin.`);
    if(data===null) throw new Error('Sunucu geçerli JSON yanıtı vermedi.');
    return data as T;
  } catch(error) {
    if(error instanceof TypeError || (error instanceof Error && error.name==='AbortError')) throw new Error('Backend bağlantısı kurulamadı veya zaman aşımına uğradı. Kaydın durumunu bağlantı yenilendiğinde kontrol edin.');
    throw error;
  } finally {clearTimeout(timer)}
}

interface ApiSubscriber {id:string;model_score:number|null;inspection_rank:number|null;data_quality_percent:number}
interface ApiState {capacity:number;plan:{subscriber_id:string}[];notes:{subscriber_id:string;note:string}[];results:{subscriber_id:string;status:Status;note:string;updated_at:string}[]}

export async function loadState():Promise<SavedState>{
  const s=await request<ApiState>('/state');
  return {capacity:s.capacity,plan:s.plan.map(p=>p.subscriber_id),notes:Object.fromEntries(s.notes.map(n=>[n.subscriber_id,n.note])),results:Object.fromEntries(s.results.map(r=>[r.subscriber_id,{status:r.status,note:r.note,date:r.updated_at}]))};
}

export async function loadSubscribers():Promise<Subscriber[]>{
  const result=await request<{items:ApiSubscriber[]}>('/subscribers?limit=100');
  return Promise.all(result.items.map(async a=>{
    const series=await request<{points:{date:string;consumption_kwh:number|null;moving_average_7d:number|null}[]}>(`/subscribers/${encodeURIComponent(a.id)}/consumption?days=365`);
    return {id:a.id,score:a.model_score,rank:a.inspection_rank,quality:a.data_quality_percent,points:series.points,
      missing:series.points.flatMap((p,i)=>p.consumption_kwh===null?[i]:[])};
  }));
}
