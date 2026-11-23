/** GridLeak AI — FastAPI backend ile tüm HTTP (GET/PUT/DELETE) iletişimini yürüten istemci katmanı. */
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
    if(!response.ok) throw new Error(typeof data?.detail==='string'?data.detail:`Ä°stek tamamlanamadÄ± (${response.status}). AlanlarÄ± kontrol edin.`);
    if(data===null) throw new Error('Sunucu geÃ§erli JSON yanÄ±tÄ± vermedi.');
    return data as T;
  } catch(error) {
    if(error instanceof TypeError || (error instanceof Error && error.name==='AbortError')) throw new Error('Backend baÄŸlantÄ±sÄ± kurulamadÄ± veya zaman aÅŸÄ±mÄ±na uÄŸradÄ±. KaydÄ±n durumunu baÄŸlantÄ± yenilendiÄŸinde kontrol edin.');
    throw error;
  } finally {clearTimeout(timer)}
}

interface ApiSubscriber {id:string;partition:string;model_score:number|null;inspection_rank:number|null;data_quality_percent:number}
interface ApiState {capacity:number;plan:{subscriber_id:string}[];notes:{subscriber_id:string;note:string}[];results:{subscriber_id:string;status:Status;note:string;updated_at:string}[]}

export async function loadState():Promise<SavedState>{
  const s=await request<ApiState>('/state');
  return {capacity:s.capacity,plan:s.plan.map(p=>p.subscriber_id),notes:Object.fromEntries(s.notes.map(n=>[n.subscriber_id,n.note])),results:Object.fromEntries(s.results.map(r=>[r.subscriber_id,{status:r.status,note:r.note,date:r.updated_at}]))};
}

export async function loadSubscribers():Promise<Subscriber[]>{
 const all:ApiSubscriber[]=[];let offset=0;
 while(true){const r=await request<{items:ApiSubscriber[];total:number}>(`/subscribers?limit=1000&offset=${offset}`);all.push(...r.items);offset+=r.items.length;if(offset>=r.total||!r.items.length)break;}
 return all.map(a=>({id:a.id,partition:a.partition,score:a.model_score,rank:a.inspection_rank,quality:a.data_quality_percent,points:[],missing:[]}));
}
export async function loadConsumption(a:Subscriber):Promise<Subscriber>{
 const r=await request<{points:Subscriber['points']}>(`/subscribers/${encodeURIComponent(a.id)}/consumption?days=365`);
 return {...a,points:r.points,missing:r.points.flatMap((p,i)=>p.consumption_kwh===null?[i]:[])};
}
