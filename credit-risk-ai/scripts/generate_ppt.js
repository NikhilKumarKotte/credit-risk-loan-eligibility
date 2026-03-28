const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');

const ROOT = path.resolve(__dirname, '..');
const OUT_PATH = path.join(ROOT, 'artifacts', 'CreditAI_Presentation.pptx');

const metricsPath = path.join(ROOT, 'models', 'training_metrics.json');
let metrics = null;
try { metrics = JSON.parse(fs.readFileSync(metricsPath, 'utf8')); } catch(e) { metrics = null; }

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';

const COLORS = {
  bg: 'FFFFFF',
  text: '1F1F1F',
  subtext: '4D4D4D',
  accent1: '4F81BD',
  accent2: 'C0504D',
  accent3: '9BBB59',
  accent4: '8064A2',
  accent5: '4BACC6',
  accent6: 'F79646',
};

function addTitle(slide, title, subtitle) {
  slide.addText(title, { x: 0.6, y: 0.6, w: 12.1, h: 0.8, fontFace: 'Calibri', fontSize: 36, color: COLORS.accent1, bold: true });
  if (subtitle) {
    slide.addText(subtitle, { x: 0.6, y: 1.6, w: 12.1, h: 0.5, fontFace: 'Calibri', fontSize: 18, color: COLORS.subtext });
  }
}

function addSectionTitle(slide, title) {
  slide.addText(title, { x: 0.6, y: 0.4, w: 12.1, h: 0.6, fontFace: 'Calibri', fontSize: 30, color: COLORS.accent1, bold: true });
}

function addBullets(slide, bullets, opts = {}) {
  const x = opts.x ?? 0.9;
  const y = opts.y ?? 1.4;
  const w = opts.w ?? 11.5;
  const h = opts.h ?? 5.5;
  slide.addText(bullets.map(t => ({ text: t, options: { bullet: { indent: 18 } } })), {
    x, y, w, h, fontFace: 'Calibri', fontSize: 18, color: COLORS.text, valign: 'top', lineSpacingMultiple: 1.2,
  });
}

function addImage(slide, imgPath, x, y, w, h) {
  if (fs.existsSync(imgPath)) {
    slide.addImage({ path: imgPath, x, y, w, h });
  }
}

// Slide 1: Title
let slide = pptx.addSlide();
addTitle(slide, 'AI Credit Risk & Loan Default Prediction Platform', 'Bank + Client Portals with Explainable AI and Live Market Signals');
slide.addText('K. Nikhil Kumar', { x: 0.6, y: 2.4, w: 12, h: 0.4, fontFace: 'Calibri', fontSize: 18, color: COLORS.text });
slide.addText('Dept of CSE (AI & ML), DRK Institute of Science and Technology', { x: 0.6, y: 2.9, w: 12, h: 0.4, fontFace: 'Calibri', fontSize: 14, color: COLORS.subtext });

// Slide 2: Abstract
slide = pptx.addSlide();
addSectionTitle(slide, 'Abstract');
addBullets(slide, [
  'End-to-end AI platform for credit risk and loan default prediction.',
  'Secure dual-portal system for banks and clients.',
  'Explainable risk scoring with model analytics and feature importance.',
  'Affordability analysis using DTI and EMI capacity.',
  'Live stock holdings integrated into client creditworthiness.',
]);

// Slide 3: Introduction
slide = pptx.addSlide();
addSectionTitle(slide, 'Introduction');
addBullets(slide, [
  'Banking requires fast, transparent, and accurate risk assessment.',
  'Traditional rule-based scores are static and opaque.',
  'Machine learning enables richer signals and explainability.',
  'CreditAI provides real-time risk evaluation with human-readable insights.',
]);

// Slide 4: Problem Statement & Motivation
slide = pptx.addSlide();
addSectionTitle(slide, 'Problem Statement & Motivation');
addBullets(slide, [
  'Loan risk evaluation needs to balance accuracy, speed, and transparency.',
  'Borrower affordability is dynamic and influenced by assets and expenses.',
  'Stakeholders need dashboards for both institutional and client views.',
  'Goal: deliver a practical, explainable AI system for lending decisions.',
]);

// Slide 5: Existing Systems & Limitations
slide = pptx.addSlide();
addSectionTitle(slide, 'Existing Systems & Limitations');
addBullets(slide, [
  'Static credit scores ignore real-time financial context.',
  'Black-box models reduce trust and auditability.',
  'Limited integration of alternative assets (e.g., stocks).',
  'User experience often lacks client-facing transparency.',
]);

// Slide 6: Proposed System
slide = pptx.addSlide();
addSectionTitle(slide, 'Proposed System');
addBullets(slide, [
  'Supervised ML pipeline with engineered financial features.',
  'Explainable risk scoring and model analytics dashboard.',
  'Secure login with role-based navigation (Bank / Client).',
  'Client portal for eligibility estimation and live stock valuation.',
  'Market Watch page for real-time stock changes.',
]);

// Slide 7: System Architecture (use dashboard screenshot)
slide = pptx.addSlide();
addSectionTitle(slide, 'System Interface (Bank Dashboard)');
addImage(slide, 'F:/pen/images/Screenshot 2026-03-16 204227.png', 0.6, 1.2, 12.1, 5.8);

// Slide 8: Client Portal + Stock Holdings
slide = pptx.addSlide();
addSectionTitle(slide, 'Client Portal & Stock Holdings');
addImage(slide, 'F:/pen/images/Screenshot 2026-03-16 203916.png', 0.6, 1.2, 12.1, 5.8);

// Slide 9: Market Watch
slide = pptx.addSlide();
addSectionTitle(slide, 'Market Watch');
addImage(slide, 'F:/pen/images/Screenshot 2026-03-16 204058.png', 0.6, 1.2, 12.1, 5.8);

// Slide 10: Eligibility Breakdown
slide = pptx.addSlide();
addSectionTitle(slide, 'Eligibility Breakdown');
addImage(slide, 'F:/pen/images/Screenshot 2026-03-16 204159.png', 0.6, 1.2, 12.1, 5.8);

// Slide 11: Applicant Evaluation Form
slide = pptx.addSlide();
addSectionTitle(slide, 'Applicant Evaluation (Bank)');
addImage(slide, 'F:/pen/images/Screenshot 2026-03-16 204338.png', 0.6, 1.2, 12.1, 5.8);

// Slide 12: Model Analytics
slide = pptx.addSlide();
addSectionTitle(slide, 'Model Analytics');
addImage(slide, 'F:/pen/images/Screenshot 2026-03-16 204259.png', 0.6, 1.2, 12.1, 5.8);

// Slide 13: Feature Importance
slide = pptx.addSlide();
addSectionTitle(slide, 'Feature Importance');
addImage(slide, 'F:/pen/images/Screenshot 2026-03-16 204322.png', 0.6, 1.2, 12.1, 5.8);

// Slide 14: Risk Output
slide = pptx.addSlide();
addSectionTitle(slide, 'Risk Output & Recommendation');
addImage(slide, 'F:/pen/images/Screenshot 2026-03-16 104016.png', 0.6, 1.2, 12.1, 5.8);

// Slide 15: Results Summary
slide = pptx.addSlide();
addSectionTitle(slide, 'Results Summary');
if (metrics) {
  const best = metrics.best_model || 'N/A';
  const rows = Object.keys(metrics)
    .filter(k => k !== 'best_model')
    .map(k => {
      const m = metrics[k];
      return `${k}: Acc ${m.accuracy?.toFixed(3)}, ROC-AUC ${m.roc_auc?.toFixed(3)}, Precision ${m.precision?.toFixed(3)}, Recall ${m.recall?.toFixed(3)}`;
    });
  addBullets(slide, [
    `Best model: ${best}.`,
    ...rows,
  ], { y: 1.4 });
} else {
  addBullets(slide, ['Model evaluation metrics available in the dashboard.']);
}

// Slide 16: Conclusion
slide = pptx.addSlide();
addSectionTitle(slide, 'Conclusion');
addBullets(slide, [
  'CreditAI delivers fast, explainable loan risk assessment.',
  'Dual-portal design supports banks and clients with tailored views.',
  'Live stock assets improve affordability estimation.',
  'System is extensible for bureau data and advanced calibration.',
]);

// Slide 17: Future Scope
slide = pptx.addSlide();
addSectionTitle(slide, 'Future Scope');
addBullets(slide, [
  'Institution-specific policy calibration and model governance.',
  'Integration of bureau scores and cash-flow analytics.',
  'Expanded market watch coverage and alternative assets.',
  'Automated model cards and audit-ready reporting.',
]);

// Slide 18: Thank You
slide = pptx.addSlide();
addTitle(slide, 'Thank You', 'Questions & Discussion');

pptx.writeFile({ fileName: OUT_PATH });
console.log('PPT generated at:', OUT_PATH);
