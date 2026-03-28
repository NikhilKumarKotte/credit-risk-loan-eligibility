const fs = require('fs');
const path = require('path');

const filePath = path.join('data','loan_data.csv');
const text = fs.readFileSync(filePath,'utf8');
const lines = text.trim().split(/\r?\n/);
const headers = lines[0].split(',');
const rows = lines.slice(1).map(l=>l.split(','));

const colIndex = name => headers.indexOf(name);
const numericCols = [
  'Age','Income','Years_Employed','Credit_History_Length','Number_of_Credit_Cards',
  'Outstanding_Loan_Amount','Debt_to_Income_Ratio','Monthly_Expenses','Previous_Defaults',
  'Loan_Amount','Loan_Term','Interest_Rate','Savings_Balance','Transaction_Volume','Late_Payments'
];

function stats(col){
  const idx = colIndex(col);
  const vals = rows.map(r=>parseFloat(r[idx])).filter(v=>!Number.isNaN(v));
  const n = vals.length;
  const mean = vals.reduce((a,b)=>a+b,0)/n;
  const sorted = vals.slice().sort((a,b)=>a-b);
  const min = sorted[0];
  const max = sorted[sorted.length-1];
  const median = sorted[Math.floor(n/2)];
  return {n, mean, min, max, median};
}

const defaultIdx = colIndex('Loan_Default');
const defaults = rows.map(r=>parseInt(r[defaultIdx],10)).filter(v=>!Number.isNaN(v));
const defaultRate = defaults.reduce((a,b)=>a+b,0)/defaults.length;

const out = { rowCount: rows.length, colCount: headers.length, defaultRate };

out.numeric = {};
numericCols.forEach(c=>{ out.numeric[c]=stats(c); });

fs.mkdirSync('artifacts',{recursive:true});
fs.writeFileSync('artifacts/dataset_stats.json', JSON.stringify(out,null,2));
console.log('rows', out.rowCount, 'cols', out.colCount, 'defaultRate', out.defaultRate);
