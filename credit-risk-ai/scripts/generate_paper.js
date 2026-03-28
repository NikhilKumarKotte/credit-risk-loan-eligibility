const fs = require('fs');
const path = require('path');
const PDFDocument = require('pdfkit');

const ROOT = path.resolve(__dirname, '..');
const OUT_PATH = path.join(ROOT, 'artifacts', 'CreditAI_Base_Paper.pdf');

const authorName = 'K. Nikhil Kumar';
const affiliation = 'Dept of CSE (AI & ML), DRK Institute of Science and Technology';

const figures = [
  { path: 'F:/pen/images/Screenshot 2026-03-16 203946.png', caption: 'Login page with secure access and portal selection.' },
  { path: 'F:/pen/images/Screenshot 2026-03-16 204227.png', caption: 'Bank dashboard home with platform highlights.' },
  { path: 'F:/pen/images/Screenshot 2026-03-16 204259.png', caption: 'Model analytics: comparison of candidate models.' },
  { path: 'F:/pen/images/Screenshot 2026-03-16 204322.png', caption: 'Feature importance for the best-performing model.' },
  { path: 'F:/pen/images/Screenshot 2026-03-16 204338.png', caption: 'Applicant evaluation form for bank users.' },
  { path: 'F:/pen/images/Screenshot 2026-03-16 203916.png', caption: 'Client portal with stock holdings input and eligibility form.' },
  { path: 'F:/pen/images/Screenshot 2026-03-16 204058.png', caption: 'Market Watch with live prices and changes.' },
  { path: 'F:/pen/images/Screenshot 2026-03-16 104016.png', caption: 'Risk assessment output with default probability and recommendation.' },
  { path: 'F:/pen/images/Screenshot 2026-03-16 204159.png', caption: 'Eligibility breakdown showing income, DTI, and EMI capacity.' },
];

const metricsPath = path.join(ROOT, 'models', 'training_metrics.json');
let metrics = null;
try {
  metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf8'));
} catch (e) {
  metrics = null;
}

const doc = new PDFDocument({ size: 'A4', margins: { top: 50, bottom: 50, left: 50, right: 50 } });
const stream = fs.createWriteStream(OUT_PATH);
doc.pipe(stream);

const pageWidth = doc.page.width - doc.page.margins.left - doc.page.margins.right;
const pageHeight = doc.page.height - doc.page.margins.top - doc.page.margins.bottom;

function ensureSpace(heightNeeded) {
  const remaining = doc.page.height - doc.y - doc.page.margins.bottom;
  if (remaining < heightNeeded) {
    doc.addPage();
  }
}

function addTitle() {
  doc.font('Helvetica-Bold').fontSize(20).text('AI Credit Risk & Loan Default Prediction Platform', { align: 'center' });
  doc.moveDown(0.5);
  doc.font('Helvetica').fontSize(12).text(authorName, { align: 'center' });
  doc.font('Helvetica').fontSize(11).text(affiliation, { align: 'center' });
  doc.moveDown(1);
}

function addSection(title, paragraphs) {
  ensureSpace(80);
  doc.font('Helvetica-Bold').fontSize(14).text(title);
  doc.moveDown(0.3);
  doc.font('Helvetica').fontSize(11);
  paragraphs.forEach(p => {
    doc.text(p, { align: 'justify' });
    doc.moveDown(0.4);
  });
}

function addFigure(fig, index) {
  const imgPath = fig.path;
  if (!fs.existsSync(imgPath)) {
    return;
  }
  const caption = `Figure ${index}: ${fig.caption}`;
  const maxWidth = pageWidth;
  const maxHeight = pageHeight * 0.55;

  ensureSpace(maxHeight + 60);
  doc.image(imgPath, { fit: [maxWidth, maxHeight], align: 'center' });
  doc.moveDown(0.3);
  doc.font('Helvetica-Oblique').fontSize(10).text(caption, { align: 'center' });
  doc.moveDown(0.8);
}

addTitle();

addSection('Abstract', [
  'This paper presents an end-to-end AI credit risk and loan default prediction platform designed for banking and retail clients. The system integrates data preprocessing, feature engineering, supervised machine learning models, explainable risk scoring, and a secure dual-portal interface for bank staff and clients. The platform extends traditional borrower assessment by incorporating behavioral indicators, affordability analysis, and live equity holdings to estimate financial resilience. The result is a practical decision-support tool that delivers fast risk insights, transparent recommendations, and portfolio-level analytics.',
]);

addSection('Keywords', [
  'Credit Risk, Loan Default Prediction, Machine Learning, Explainable AI, Streamlit, Risk Scoring, Portfolio Analytics, Market Data Integration.',
]);

addSection('1. Introduction', [
  'Banking institutions require accurate and explainable tools to evaluate loan risk under evolving economic conditions. Conventional rules-based scoring can be opaque or overly static, while modern lending demands real-time analytics, richer feature signals, and user-friendly decision support. The proposed CreditAI platform addresses these needs by combining supervised learning with engineered financial indicators, interactive dashboards, and explainability visualizations.',
  'The paper structure and presentation are inspired by the professional format observed in a recent Random Forest research article, adapted to the credit risk domain. We describe the system architecture, methodology, interface, and results, and then discuss limitations and future scope.',
]);

addSection('2. System Overview', [
  'The platform is implemented as a Streamlit application with two secure portals: (i) Bank Dashboard for applicant evaluation and model analytics, and (ii) Client Portal for eligibility estimation and personal risk insights. The system includes a live Market Watch module to display top stock prices and change percentages, and a stock holdings component that augments borrower assets during scoring.',
]);

addSection('3. Methodology', [
  'The system follows a standard supervised learning pipeline: data ingestion, preprocessing, feature engineering, model training, evaluation, and deployment. Key borrower features include income, loan terms, outstanding obligations, credit history length, payment behavior, and spending indicators. Engineered features capture credit utilization, repayment stability, and loan burden. A scoring engine converts predicted default probability into a credit score range with risk tier classification.',
]);

addSection('4. Model Training and Evaluation', [
  'Three models were evaluated: Logistic Regression, Random Forest, and XGBoost. The best-performing model is selected based on ROC-AUC and overall stability. Performance metrics are persisted and visualized in the Model Analytics dashboard.',
]);

if (metrics) {
  const best = metrics.best_model || 'N/A';
  const summary = Object.keys(metrics)
    .filter(k => k !== 'best_model')
    .map(k => {
      const m = metrics[k];
      return `${k}: Accuracy ${m.accuracy?.toFixed(4)}, ROC-AUC ${m.roc_auc?.toFixed(4)}, Precision ${m.precision?.toFixed(4)}, Recall ${m.recall?.toFixed(4)}`;
    });
  addSection('4.1 Metrics Summary', [
    `Best model: ${best}.`,
    ...summary,
  ]);
}

addSection('5. Client-Side Eligibility and Live Asset Integration', [
  'Clients can evaluate how much loan they can afford based on a DTI-based capacity model that compares monthly income against expenses and estimated EMI. The system also accepts equity holdings (ticker and shares) and uses live prices to compute portfolio value. This value is integrated into the applicant assets, enabling a more realistic assessment of repayment capacity.',
  'A Market Watch page displays live prices and changes for major equities, enabling users to see how market movement may affect their asset-based eligibility.',
]);

addSection('6. Results and Interface Outputs', [
  'The following figures demonstrate key system outputs: secure login, bank home dashboard, model analytics, feature importance visualization, applicant evaluation inputs, client portal with stock holdings, market watch, and risk assessment results.',
]);

let figIndex = 1;
figures.forEach(fig => addFigure(fig, figIndex++));

addSection('7. Discussion', [
  'The platform delivers explainable, fast credit insights while supporting both institutional evaluation and individual eligibility checks. Feature importance and risk-tier explanations improve transparency, and the integration of live assets introduces dynamic signals that are often missing from static scoring systems. The approach also enables a smoother customer experience through a client portal, while retaining rigorous evaluation controls in the bank dashboard.',
]);

addSection('8. Conclusion and Future Scope', [
  'CreditAI demonstrates a practical, deployable AI-driven credit risk solution that blends machine learning, explainability, and live financial signals. Future work includes model calibration for institution-specific policies, incorporation of bureau data, and richer behavioral signals such as transactional timelines and cash-flow stability. Additional work may also include automated decision thresholds and audit-ready model cards.',
]);

addSection('References', [
  'Paithane, P. M. (2023). Random Forest Algorithm Use for Crop Recommendation. Journal of Engineering and Technology for Industrial Applications, 9(43), 34–41.',
  'Additional internal project documentation and model artifacts from the CreditAI platform (2026).',
]);

doc.end();

stream.on('finish', () => {
  console.log('PDF generated at:', OUT_PATH);
});
