/** GridLeak AI — React uygulama giriş noktası; App bileşenini kök DOM düğümüne bağlar. */
import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles.css';
createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>);
