/** GridLeak AI — Ortak TypeScript veri tipleri, ekran tanımları, sayı formatlama ve güvenli CSV dışa aktarım yardımcıları. */
export type Page = 'review' | 'plan' | 'results';
export const pages: {id: Page; label: string; title: string; subtitle: string}[] = [
  {id:'review',label:'İnceleme merkezi',title:'Abone incelemesi',subtitle:'Tüketim davranışı, veri yeterliliği ve açıklanabilir denetim önceliği'},
  {id:'plan',label:'Denetim planı',title:'Denetim planı',subtitle:'Kapasiteye göre aday seçimi ve görev takibi'},
  {id:'results',label:'Saha sonuçları',title:'Saha sonuçları',subtitle:'Denetim sonrasında bulguları kaydedin ve takip edin'},
];
export interface Subscriber { id:string; partition?:string; score:number|null; rank:number|null; quality:number; missing:number[]; points:{date:string;consumption_kwh:number|null;moving_average_7d:number|null}[] }
export const nf = (n:number) => new Intl.NumberFormat('tr-TR',{maximumFractionDigits:1}).format(n);
export const quality = (a:Subscriber) => a.quality;
export const statuses = ['Planlandı','İnceleme sürüyor','Kaçak doğrulandı','Kaçak bulunmadı','Ek inceleme gerekli'] as const;
export type Status = typeof statuses[number];
export type Result = {status:Status; note:string; date:string};
export interface SavedState { plan:string[]; notes:Record<string,string>; results:Record<string,Result>; capacity:number }
export function consumption(a:Subscriber){return a.points.map(p=>p.consumption_kwh)}
export function downloadCsv(name:string, rows:(string|number)[][]){
  const escape=(v:string|number)=>'"'+String(typeof v==='string'&&/^[=+@\-\t\r]/.test(v)?"'"+v:v).replaceAll('"','""')+'"';
  const url=URL.createObjectURL(new Blob(['\uFEFF'+rows.map(r=>r.map(escape).join(';')).join('\r\n')],{type:'text/csv;charset=utf-8'}));
  const a=document.createElement('a');a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
}