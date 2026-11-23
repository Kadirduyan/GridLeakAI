/** Üst Gezinme Menüsü — İnceleme Merkezi, Denetim Planı, Saha Sonuçları ve Model/Veri ekranları arasında geçiş sağlar. */
import {pages,type Page} from '../data';
export default function Menu({page,onChange}:{page:Page;onChange:(p:Page)=>void}){
  return <nav className="gl-nav" aria-label="Ana menü">{pages.map(p=><a key={p.id} href={'#/'+p.id} aria-current={page===p.id?'page':undefined} onClick={e=>{e.preventDefault();onChange(p.id)}}>{p.label}</a>)}</nav>;
}
