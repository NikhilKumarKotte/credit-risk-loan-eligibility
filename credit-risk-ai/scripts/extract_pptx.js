const fs = require('fs');
const JSZip = require('jszip');

async function readTheme(pptxPath){
  const data = fs.readFileSync(pptxPath);
  const zip = await JSZip.loadAsync(data);
  const themeFile = zip.file('ppt/theme/theme1.xml');
  if(!themeFile){
    return null;
  }
  const xml = await themeFile.async('string');
  return xml;
}

async function readSlideTexts(pptxPath){
  const data = fs.readFileSync(pptxPath);
  const zip = await JSZip.loadAsync(data);
  const slides = Object.keys(zip.files).filter(f => f.startsWith('ppt/slides/slide') && f.endsWith('.xml'));
  slides.sort((a,b)=>{
    const na = parseInt(a.match(/slide(\d+)/)[1],10);
    const nb = parseInt(b.match(/slide(\d+)/)[1],10);
    return na-nb;
  });
  const results = [];
  for(const s of slides){
    const xml = await zip.file(s).async('string');
    const texts = [];
    const regex = /<a:t>(.*?)<\/a:t>/g;
    let m;
    while((m = regex.exec(xml))){
      const t = m[1].replace(/&amp;/g,'&').replace(/&lt;/g,'<').replace(/&gt;/g,'>');
      if(t.trim()) texts.push(t.trim());
    }
    results.push({slide: s, texts});
  }
  return results;
}

(async ()=>{
  const themeXml = await readTheme('C:/Users/Lenovo/Downloads/Automated Compliance Checker for Legal Metrology Declarations on E-Commerce Platforms.pptx');
  fs.writeFileSync('artifacts/theme1.xml', themeXml || '');
  const slides = await readSlideTexts('C:/Users/Lenovo/Downloads/Stock-Price-Prediction-Using-LSTM-Networks1[1].pptx');
  fs.writeFileSync('artifacts/stock_ref_text.json', JSON.stringify(slides, null, 2));
  console.log('theme xml length', themeXml ? themeXml.length : 0);
  console.log('slides', slides.length);
})();
