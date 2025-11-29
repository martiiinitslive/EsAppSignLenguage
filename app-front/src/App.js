import React from 'react';
import MainTranslator from './components/MainTranslator';
import logo from './logo.svg';
import './App.css';

function App() {
  return (
    <div className="App">
      <header>
        <h1>EsApp Sign Language AI</h1>
        <p>Multimedia translator: text, audio, video or link</p>
      </header>
      <MainTranslator />
    </div>
  );
}

export default App;
