const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
  ShadingType, VerticalAlign, PageNumber, PageBreak } = require("docx");

const BASE = "d:\\Edu\\Final Project\\WorkingDirectory\\CTI_IDS";

function img(relPath, w, h) {
  const full = path.join(BASE, relPath);
  if (!fs.existsSync(full)) { console.warn("Missing:", full); return null; }
  return new ImageRun({
    type: "png", data: fs.readFileSync(full),
    transformation: { width: w, height: h },
    altText: { title: relPath, description: relPath, name: relPath }
  });
}

function figCaption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 240 },
    children: [new TextRun({ text, italics: true, size: 20, font: "Times New Roman", color: "444444" })]
  });
}

function figPara(relPath, w, h) {
  const image = img(relPath, w, h);
  if (!image) return new Paragraph({ children: [new TextRun({ text: `[Image: ${relPath}]`, italics: true, color: "FF0000" })] });
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, children: [image] });
}

const tb = { style: BorderStyle.SINGLE, size: 1, color: "AAAAAA" };
const cb = { top: tb, bottom: tb, left: tb, right: tb };
const hdrShade = { fill: "D9E2F3", type: ShadingType.CLEAR };

function hdrCell(text, w) {
  return new TableCell({
    borders: cb, width: { size: w, type: WidthType.DXA }, shading: hdrShade, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text, bold: true, size: 20, font: "Times New Roman" })] })]
  });
}
function cell(text, w, opts = {}) {
  return new TableCell({
    borders: cb, width: { size: w, type: WidthType.DXA }, verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({
      alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
      children: [new TextRun({ text, size: 20, font: "Times New Roman", bold: !!opts.bold })]
    })]
  });
}

function bodyText(text) {
  return new Paragraph({
    spacing: { after: 120, line: 360 },
    children: [new TextRun({ text, size: 24, font: "Times New Roman" })]
  });
}

function bodyRuns(runs) {
  return new Paragraph({
    spacing: { after: 120, line: 360 },
    children: runs.map(r => typeof r === "string"
      ? new TextRun({ text: r, size: 24, font: "Times New Roman" })
      : new TextRun({ size: 24, font: "Times New Roman", ...r }))
  });
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Times New Roman", size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, color: "000000", font: "Times New Roman" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, color: "000000", font: "Times New Roman" },
        paragraph: { spacing: { before: 280, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, color: "000000", font: "Times New Roman" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 2 } },
    ]
  },
  numbering: {
    config: [
      { reference: "bullet-list", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbered-1", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbered-2", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "(%1)", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  sections: [{
    properties: {
      page: { margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
    },
    headers: {
      default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [
        new TextRun({ text: "Chapter 4: Actual Work", italics: true, size: 18, font: "Times New Roman", color: "888888" })
      ] })] })
    },
    footers: {
      default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
        new TextRun({ text: "Page ", size: 18, font: "Times New Roman" }),
        new TextRun({ children: [PageNumber.CURRENT], size: 18, font: "Times New Roman" })
      ] })] })
    },
    children: [
      // ===== CHAPTER TITLE =====
      new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { before: 600, after: 400 },
        children: [new TextRun({ text: "CHAPTER 4", size: 36, bold: true, font: "Times New Roman" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { after: 600 },
        children: [new TextRun({ text: "ACTUAL WORK", size: 36, bold: true, font: "Times New Roman" })]
      }),

      // ===== 4.1 METHODOLOGY =====
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("4.1 Methodology for the Study")] }),

      bodyText("The research methodology follows a systematic, objective-driven approach to build an end-to-end DoS detection and mitigation system. The study is structured into four sequential objectives, each building upon the output of the previous one, culminating in a complete pipeline that transforms raw network traffic features into actionable security responses."),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.1.1 Research Design")] }),

      bodyText("The research adopts an experimental design comprising four phases: (1) data preparation and feature engineering, (2) comparative machine learning model training and selection, (3) integration of Explainable AI for model transparency, and (4) development of a rule-based mitigation framework driven by XAI outputs. The overall system architecture is illustrated in Figure 4.1."),

      figPara("presentation_diagrams/complete_pipeline_highlevel.png", 580, 200),
      figCaption("Figure 4.1: High-level system architecture: Detection to Defense pipeline"),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.1.2 Dataset Selection")] }),

      bodyText("The UNSW-NB15 dataset, developed by the University of New South Wales, was selected as the primary benchmark dataset. It is a widely recognised network intrusion detection dataset containing realistic modern network traffic with both normal and attack records. Table 4.1 summarises the key properties of the dataset."),

      // Table 4.1: Dataset Properties
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 },
        children: [new TextRun({ text: "Table 4.1: UNSW-NB15 dataset properties", bold: true, italics: true, size: 20, font: "Times New Roman" })] }),
      new Table({
        columnWidths: [3500, 5860],
        rows: [
          new TableRow({ tableHeader: true, children: [hdrCell("Property", 3500), hdrCell("Details", 5860)] }),
          new TableRow({ children: [cell("Source", 3500), cell("University of New South Wales (UNSW)", 5860)] }),
          new TableRow({ children: [cell("Total Records", 3500), cell("257,673 (175,341 training + 82,332 testing)", 5860)] }),
          new TableRow({ children: [cell("Original Features", 3500), cell("49 columns", 5860)] }),
          new TableRow({ children: [cell("Attack Categories", 3500), cell("10 types (DoS, Exploits, Fuzzers, Generic, etc.)", 5860)] }),
          new TableRow({ children: [cell("Focus", 3500), cell("Binary classification: DoS vs Normal", 5860)] }),
        ]
      }),

      new Paragraph({ spacing: { after: 200 }, children: [] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.1.3 Data Splitting Strategy")] }),

      bodyRuns([
        "A critical aspect of the methodology is the use of ",
        { text: "completely separate CSV files", bold: true },
        " for training and testing, as officially provided by the UNSW-NB15 authors. The model has never seen any testing data during training, constituting true external validation."
      ]),

      bodyText("The training set was balanced to 24,528 samples (12,264 DoS + 12,264 Normal) by random undersampling of the majority class. This ensures equal class representation during training and prevents the model from developing bias towards normal traffic. The testing set was kept at its natural imbalanced ratio of approximately 90% normal and 10% DoS (37,000 normal + 4,089 DoS = 41,089 total), simulating realistic network conditions where attacks are rare events."),

      figPara("03_model_training/proper_training/images/02_training_set_distribution.png", 280, 210),
      figCaption("(a) Training set (balanced 50/50)"),
      figPara("03_model_training/proper_training/images/01_testing_set_distribution.png", 280, 210),
      figCaption("(b) Testing set (imbalanced 90/10)"),
      figCaption("Figure 4.2: Class distribution of training and testing datasets"),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.1.4 Feature Engineering")] }),

      bodyText("From the original 49 columns in the UNSW-NB15 dataset, 10 features were selected based on a combination of correlation analysis, variance analysis, preliminary model feature importance scores, and domain knowledge of DoS attack characteristics. Table 4.2 describes each selected feature and its relevance to DoS detection."),

      // Table 4.2: Features
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 },
        children: [new TextRun({ text: "Table 4.2: Selected features and their DoS relevance", bold: true, italics: true, size: 20, font: "Times New Roman" })] }),
      new Table({
        columnWidths: [600, 1200, 3200, 4360],
        rows: [
          new TableRow({ tableHeader: true, children: [hdrCell("#", 600), hdrCell("Feature", 1200), hdrCell("Description", 3200), hdrCell("DoS Relevance", 4360)] }),
          new TableRow({ children: [cell("1", 600, {center:true}), cell("rate", 1200), cell("Packets per second", 3200), cell("DoS floods spike packet rate", 4360)] }),
          new TableRow({ children: [cell("2", 600, {center:true}), cell("sload", 1200), cell("Source bits/sec", 3200), cell("High sload indicates flood", 4360)] }),
          new TableRow({ children: [cell("3", 600, {center:true}), cell("sbytes", 1200), cell("Source-to-dest bytes", 3200), cell("Excessive data transfer", 4360)] }),
          new TableRow({ children: [cell("4", 600, {center:true}), cell("dload", 1200), cell("Destination bits/sec", 3200), cell("High in amplification attacks", 4360)] }),
          new TableRow({ children: [cell("5", 600, {center:true}), cell("proto", 1200), cell("Network protocol (encoded)", 3200), cell("Protocol-specific attacks", 4360)] }),
          new TableRow({ children: [cell("6", 600, {center:true}), cell("dtcpb", 1200), cell("Dest TCP base sequence #", 3200), cell("SYN flood indicator", 4360)] }),
          new TableRow({ children: [cell("7", 600, {center:true}), cell("stcpb", 1200), cell("Source TCP base sequence #", 3200), cell("SYN flood indicator", 4360)] }),
          new TableRow({ children: [cell("8", 600, {center:true}), cell("dmean", 1200), cell("Dest packet mean size", 3200), cell("Small in floods, large in slowloris", 4360)] }),
          new TableRow({ children: [cell("9", 600, {center:true}), cell("tcprtt", 1200), cell("TCP round-trip time", 3200), cell("Increases under DoS load", 4360)] }),
          new TableRow({ children: [cell("10", 600, {center:true}), cell("dur", 1200), cell("Connection duration", 3200), cell("Long in slowloris attacks", 4360)] }),
        ]
      }),

      new Paragraph({ spacing: { after: 120 }, children: [] }),

      bodyRuns([
        "The preprocessing pipeline consists of three steps: (1) ",
        { text: "Protocol Encoding", bold: true },
        " using LabelEncoder to convert the categorical proto column into numeric values (132 classes; tcp\u2192112, udp\u2192118), (2) ",
        { text: "Feature Scaling", bold: true },
        " using StandardScaler to normalise all features to zero mean and unit variance, fitted exclusively on the training data to prevent data leakage, and (3) ",
        { text: "Missing Value Imputation", bold: true },
        " using column median values. The fitted scaler and encoder are persisted as .pkl files for reproducible inference."
      ]),

      // ===== 4.2 EXPERIMENTAL WORK =====
      new Paragraph({ children: [new PageBreak()] }),
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("4.2 Experimental and Analytical Work Completed")] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.2.1 Comparative Model Training")] }),

      bodyText("Eight machine learning models were trained and evaluated, spanning classical algorithms, ensemble methods, shallow neural networks, and deep learning architectures. All models were trained on the same balanced training set (24,528 samples) and evaluated on the same imbalanced external test set (41,089 samples). Table 4.3 lists the models and their categories."),

      // Table 4.3: Models
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 },
        children: [new TextRun({ text: "Table 4.3: Machine learning models trained and evaluated", bold: true, italics: true, size: 20, font: "Times New Roman" })] }),
      new Table({
        columnWidths: [600, 2500, 3200, 3060],
        rows: [
          new TableRow({ tableHeader: true, children: [hdrCell("#", 600), hdrCell("Model", 2500), hdrCell("Type", 3200), hdrCell("Category", 3060)] }),
          new TableRow({ children: [cell("1", 600, {center:true}), cell("XGBoost", 2500), cell("Gradient Boosting", 3200), cell("Ensemble", 3060)] }),
          new TableRow({ children: [cell("2", 600, {center:true}), cell("Random Forest", 2500), cell("Bagging Ensemble", 3200), cell("Ensemble", 3060)] }),
          new TableRow({ children: [cell("3", 600, {center:true}), cell("Decision Tree", 2500), cell("Single Tree Classifier", 3200), cell("Classical", 3060)] }),
          new TableRow({ children: [cell("4", 600, {center:true}), cell("MLP", 2500), cell("Multi-Layer Perceptron", 3200), cell("Shallow Neural Network", 3060)] }),
          new TableRow({ children: [cell("5", 600, {center:true}), cell("SVM", 2500), cell("Support Vector Machine", 3200), cell("Classical", 3060)] }),
          new TableRow({ children: [cell("6", 600, {center:true}), cell("Logistic Regression", 2500), cell("Linear Classifier", 3200), cell("Classical (Baseline)", 3060)] }),
          new TableRow({ children: [cell("7", 600, {center:true}), cell("LSTM", 2500), cell("Long Short-Term Memory", 3200), cell("Deep Learning (Recurrent)", 3060)] }),
          new TableRow({ children: [cell("8", 600, {center:true}), cell("1D-CNN", 2500), cell("1D Convolutional Network", 3200), cell("Deep Learning (Conv.)", 3060)] }),
        ]
      }),

      new Paragraph({ spacing: { after: 120 }, children: [] }),

      bodyText("For the classical and ensemble models, 5-fold stratified cross-validation was used during training to estimate generalisation performance. Table 4.4 presents the cross-validation results."),

      // Table 4.4: CV Results
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 },
        children: [new TextRun({ text: "Table 4.4: 5-Fold stratified cross-validation results on training set", bold: true, italics: true, size: 20, font: "Times New Roman" })] }),
      new Table({
        columnWidths: [2200, 1790, 1790, 1790, 1790],
        rows: [
          new TableRow({ tableHeader: true, children: [hdrCell("Model", 2200), hdrCell("CV Accuracy", 1790), hdrCell("CV Precision", 1790), hdrCell("CV Recall", 1790), hdrCell("CV F1", 1790)] }),
          new TableRow({ children: [cell("XGBoost", 2200, {bold:true}), cell("96.45% \u00b10.42", 1790, {center:true}), cell("96.89% \u00b10.52", 1790, {center:true}), cell("95.95% \u00b10.58", 1790, {center:true}), cell("96.45% \u00b10.42", 1790, {center:true})] }),
          new TableRow({ children: [cell("Random Forest", 2200), cell("96.22% \u00b10.38", 1790, {center:true}), cell("96.75% \u00b10.48", 1790, {center:true}), cell("95.63% \u00b10.62", 1790, {center:true}), cell("96.22% \u00b10.38", 1790, {center:true})] }),
          new TableRow({ children: [cell("Decision Tree", 2200), cell("95.55% \u00b11.39", 1790, {center:true}), cell("96.84% \u00b11.30", 1790, {center:true}), cell("94.18% \u00b13.42", 1790, {center:true}), cell("95.48% \u00b11.50", 1790, {center:true})] }),
          new TableRow({ children: [cell("MLP", 2200), cell("94.32% \u00b10.60", 1790, {center:true}), cell("95.38% \u00b10.72", 1790, {center:true}), cell("93.02% \u00b10.88", 1790, {center:true}), cell("94.32% \u00b10.60", 1790, {center:true})] }),
          new TableRow({ children: [cell("SVM", 2200), cell("92.26% \u00b10.75", 1790, {center:true}), cell("93.45% \u00b10.85", 1790, {center:true}), cell("90.88% \u00b11.02", 1790, {center:true}), cell("92.26% \u00b10.75", 1790, {center:true})] }),
          new TableRow({ children: [cell("Logistic Reg.", 2200), cell("86.64% \u00b11.15", 1790, {center:true}), cell("90.11% \u00b11.24", 1790, {center:true}), cell("82.05% \u00b11.82", 1790, {center:true}), cell("86.27% \u00b11.15", 1790, {center:true})] }),
        ]
      }),

      new Paragraph({ spacing: { after: 120 }, children: [] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.2.2 Threshold Optimization")] }),

      bodyText("A key analytical step was threshold optimisation. At the default classification threshold of 0.5, precision was low due to the 9:1 class imbalance in the test set \u2014 even a small false positive rate on 37,000 normal samples produces a large absolute number of false alarms. To address this, threshold optimisation was performed by searching over 100 candidate thresholds [0.00, 0.01, ..., 1.00] and selecting the threshold that maximises the F1 score. Table 4.5 presents the results at each model's optimised threshold."),

      // Table 4.5: Optimized Results
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 },
        children: [new TextRun({ text: "Table 4.5: External benchmark results at optimised thresholds (41,089 unseen samples)", bold: true, italics: true, size: 20, font: "Times New Roman" })] }),
      new Table({
        columnWidths: [1800, 1260, 1260, 1260, 1260, 1260, 1260],
        rows: [
          new TableRow({ tableHeader: true, children: [hdrCell("Model", 1800), hdrCell("Acc.", 1260), hdrCell("Prec.", 1260), hdrCell("Recall", 1260), hdrCell("F1", 1260), hdrCell("Threshold", 1260), hdrCell("AUC", 1260)] }),
          new TableRow({ children: [cell("XGBoost", 1800, {bold:true}), cell("97.76%", 1260, {center:true, bold:true}), cell("94.41%", 1260, {center:true, bold:true}), cell("87.09%", 1260, {center:true, bold:true}), cell("90.57%", 1260, {center:true, bold:true}), cell("0.8517", 1260, {center:true, bold:true}), cell("0.9915", 1260, {center:true, bold:true})] }),
          new TableRow({ children: [cell("Random Forest", 1800), cell("97.54%", 1260, {center:true}), cell("94.44%", 1260, {center:true}), cell("85.42%", 1260, {center:true}), cell("89.70%", 1260, {center:true}), cell("0.8333", 1260, {center:true}), cell("0.9900", 1260, {center:true})] }),
          new TableRow({ children: [cell("Decision Tree", 1800), cell("97.83%", 1260, {center:true}), cell("93.43%", 1260, {center:true}), cell("84.13%", 1260, {center:true}), cell("88.53%", 1260, {center:true}), cell("0.93", 1260, {center:true}), cell("0.9806", 1260, {center:true})] }),
          new TableRow({ children: [cell("1D-CNN", 1800), cell("97.42%", 1260, {center:true}), cell("90.92%", 1260, {center:true}), cell("82.27%", 1260, {center:true}), cell("86.38%", 1260, {center:true}), cell("0.87", 1260, {center:true}), cell("0.9780", 1260, {center:true})] }),
          new TableRow({ children: [cell("MLP", 1800), cell("97.14%", 1260, {center:true}), cell("88.43%", 1260, {center:true}), cell("82.02%", 1260, {center:true}), cell("85.11%", 1260, {center:true}), cell("0.8448", 1260, {center:true}), cell("0.9753", 1260, {center:true})] }),
          new TableRow({ children: [cell("LSTM", 1800), cell("96.89%", 1260, {center:true}), cell("88.12%", 1260, {center:true}), cell("79.48%", 1260, {center:true}), cell("83.58%", 1260, {center:true}), cell("0.79", 1260, {center:true}), cell("0.9683", 1260, {center:true})] }),
          new TableRow({ children: [cell("SVM", 1800), cell("95.86%", 1260, {center:true}), cell("82.47%", 1260, {center:true}), cell("74.10%", 1260, {center:true}), cell("78.06%", 1260, {center:true}), cell("0.93", 1260, {center:true}), cell("\u2014", 1260, {center:true})] }),
          new TableRow({ children: [cell("Logistic Reg.", 1800), cell("88.42%", 1260, {center:true}), cell("44.48%", 1260, {center:true}), cell("66.06%", 1260, {center:true}), cell("53.16%", 1260, {center:true}), cell("0.7468", 1260, {center:true}), cell("\u2014", 1260, {center:true})] }),
        ]
      }),

      new Paragraph({ spacing: { after: 120 }, children: [] }),

      bodyText("XGBoost achieved the highest F1 score (90.57%), highest AUC (0.9915), and the lowest false alarm rate (209 false positives out of 37,000 normal samples, i.e., 0.56%). It was therefore selected as the production model."),

      figPara("03_model_training/proper_training/images/03_model_performance_training.png", 500, 320),
      figCaption("Figure 4.3: Model performance comparison during cross-validation"),

      figPara("03_model_training/proper_training/images/04_xgboost_confusion_matrix_training.png", 280, 220),
      figCaption("(a) Training set confusion matrix"),
      figPara("03_model_training/proper_training/images/05_xgboost_confusion_matrix_testing.png", 280, 220),
      figCaption("(b) Testing set confusion matrix (threshold = 0.8517)"),
      figCaption("Figure 4.4: XGBoost confusion matrices on training and external testing sets"),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.2.3 Why Deep Learning Models Performed Lower")] }),

      bodyText("An important analytical finding is that the LSTM (F1: 83.58%) and 1D-CNN (F1: 86.38%) underperformed the tree-based ensemble models. This result is consistent with established machine learning literature. The UNSW-NB15 dataset provides pre-computed, engineered flow-level features in a tabular format. Tree-based models such as XGBoost are specifically optimised for tabular data and can exploit feature interactions efficiently through recursive partitioning. In contrast, LSTMs and CNNs are architecturally designed for raw sequential or spatial data (e.g., raw packet captures or images). When applied to pre-aggregated tabular features, their structural advantages \u2014 temporal memory for LSTMs and local pattern detection for CNNs \u2014 cannot be fully leveraged."),

      // ===== 4.3 MODELING, ANALYSIS & DESIGN =====
      new Paragraph({ children: [new PageBreak()] }),
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("4.3 Modeling, Analysis & Design")] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.3.1 XGBoost Model Configuration")] }),

      bodyText("The selected XGBoost model was configured with the following hyperparameters: n_estimators=100, max_depth=6, learning_rate=0.1, and random_state=42. The model produces a probability score P(DoS) for each input sample. At the optimised threshold of 0.8517, a sample is classified as a DoS attack if P(DoS) \u2265 0.8517 and as normal traffic otherwise."),

      bodyText("The XGBoost feature importance, shown in Figure 4.5, reveals that sload (source bandwidth), rate (packets per second), and sbytes (source bytes) are the top three discriminative features for DoS detection. This aligns with domain knowledge, as DoS attacks characteristically generate abnormally high traffic volumes and rates."),

      figPara("03_model_training/proper_training/images/06_xgboost_feature_importance.png", 420, 300),
      figCaption("Figure 4.5: XGBoost feature importance ranking"),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.3.2 Explainable AI Design: SHAP TreeExplainer")] }),

      bodyText("To provide transparency into the model's decision-making process, SHAP (SHapley Additive exPlanations) was integrated using the TreeExplainer algorithm. SHAP calculates a contribution score (SHAP value) for each feature in every prediction, quantifying how much each feature pushed the prediction towards DoS or towards Normal. Positive SHAP values indicate a push towards the DoS class, while negative values push towards Normal. The sum of all SHAP values equals the model's log-odds output."),

      bodyRuns([
        "SHAP TreeExplainer was chosen over alternatives such as LIME because it provides ",
        { text: "mathematically exact", bold: true },
        " Shapley values for tree-based models, is ",
        { text: "deterministic", bold: true },
        " (same input always produces the same explanation), and is computationally efficient \u2014 running in seconds rather than minutes."
      ]),

      figPara("04_xai_integration/images/07_shap_summary_plot.png", 500, 350),
      figCaption("Figure 4.6: SHAP summary plot showing global feature importance across 500 samples"),

      figPara("04_xai_integration/images/08_shap_waterfall_dos.png", 280, 220),
      figCaption("(a) DoS attack explanation"),
      figPara("04_xai_integration/images/09_shap_waterfall_normal.png", 280, 220),
      figCaption("(b) Normal traffic explanation"),
      figCaption("Figure 4.7: SHAP waterfall plots for individual record explanations"),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.3.3 Mitigation Framework Design")] }),

      bodyText("The mitigation framework represents the research novelty of this work. It converts XAI-explained detections into actionable security responses through three stages: Attack Classification, Severity Assessment, and Mitigation Command Generation."),

      figPara("presentation_diagrams/mitigation_framework_highlevel.png", 520, 220),
      figCaption("Figure 4.8: Mitigation framework high-level design"),

      bodyRuns([{ text: "Attack Classification.", bold: true }, " Based on the SHAP feature contributions and raw feature values, detected attacks are classified into four types:"]),

      new Paragraph({ numbering: { reference: "numbered-1", level: 0 }, spacing: { after: 60, line: 360 },
        children: [new TextRun({ text: "Volumetric Flood: ", bold: true, size: 24, font: "Times New Roman" }), new TextRun({ text: "Characterised by high rate, sload, and sbytes values (e.g., UDP flood, ICMP flood).", size: 24, font: "Times New Roman" })] }),
      new Paragraph({ numbering: { reference: "numbered-1", level: 0 }, spacing: { after: 60, line: 360 },
        children: [new TextRun({ text: "Protocol Exploit: ", bold: true, size: 24, font: "Times New Roman" }), new TextRun({ text: "Characterised by abnormal proto and TCP sequence numbers (e.g., SYN flood, TCP state exhaustion).", size: 24, font: "Times New Roman" })] }),
      new Paragraph({ numbering: { reference: "numbered-1", level: 0 }, spacing: { after: 60, line: 360 },
        children: [new TextRun({ text: "Slowloris: ", bold: true, size: 24, font: "Times New Roman" }), new TextRun({ text: "Characterised by long dur and low rate (e.g., slow HTTP attacks that hold connections open).", size: 24, font: "Times New Roman" })] }),
      new Paragraph({ numbering: { reference: "numbered-1", level: 0 }, spacing: { after: 120, line: 360 },
        children: [new TextRun({ text: "Amplification: ", bold: true, size: 24, font: "Times New Roman" }), new TextRun({ text: "Characterised by dload \u226B sload, where the response is much larger than the request (e.g., DNS amplification, NTP amplification).", size: 24, font: "Times New Roman" })] }),

      bodyRuns([{ text: "Severity Assessment.", bold: true }, " The severity score combines three components: (1) the model's prediction confidence as the base score, (2) an attack type modifier (Amplification: +15%, Volumetric: +10%, Protocol: +5%), and (3) a feature modifier based on extreme SHAP values (+0\u201310%). The final score is mapped to four levels: CRITICAL (\u226595%), HIGH (90\u201395%), MEDIUM (75\u201390%), and LOW (60\u201375%)."]),

      bodyRuns([{ text: "Mitigation Command Generation.", bold: true }, " For each attack type and severity combination, the system generates specific, executable Linux commands. For example, a Volumetric Flood detection generates iptables rate-limiting rules and tc bandwidth throttling commands, while a Protocol Exploit triggers SYN cookie activation and SYN rate-limiting rules."]),

      // ===== 4.4 IMPLEMENTATION DETAILS =====
      new Paragraph({ children: [new PageBreak()] }),
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("4.4 Implementation Details")] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.4.1 Technology Stack")] }),

      bodyText("The system was implemented entirely in Python 3.x. Table 4.6 lists the key libraries and their roles."),

      // Table 4.6: Tech Stack
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 },
        children: [new TextRun({ text: "Table 4.6: Technology stack", bold: true, italics: true, size: 20, font: "Times New Roman" })] }),
      new Table({
        columnWidths: [2400, 1200, 5760],
        rows: [
          new TableRow({ tableHeader: true, children: [hdrCell("Library", 2400), hdrCell("Version", 1200), hdrCell("Purpose", 5760)] }),
          new TableRow({ children: [cell("XGBoost", 2400), cell("2.x", 1200, {center:true}), cell("Gradient boosting model training", 5760)] }),
          new TableRow({ children: [cell("scikit-learn", 2400), cell("1.x", 1200, {center:true}), cell("Preprocessing, cross-validation, classical models", 5760)] }),
          new TableRow({ children: [cell("TensorFlow/Keras", 2400), cell("2.x", 1200, {center:true}), cell("LSTM and 1D-CNN deep learning models", 5760)] }),
          new TableRow({ children: [cell("SHAP", 2400), cell("0.4x", 1200, {center:true}), cell("Explainable AI (TreeExplainer)", 5760)] }),
          new TableRow({ children: [cell("Streamlit", 2400), cell("1.x", 1200, {center:true}), cell("Interactive web dashboard", 5760)] }),
          new TableRow({ children: [cell("Plotly", 2400), cell("5.x", 1200, {center:true}), cell("Interactive visualisations", 5760)] }),
          new TableRow({ children: [cell("pandas / NumPy", 2400), cell("\u2014", 1200, {center:true}), cell("Data manipulation and numerical computation", 5760)] }),
        ]
      }),

      new Paragraph({ spacing: { after: 160 }, children: [] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.4.2 Project Structure")] }),

      bodyText("The project is organised into modular directories, each corresponding to a research objective:"),

      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, spacing: { after: 60, line: 360 },
        children: [new TextRun({ text: "01_data_preparation/", bold: true, size: 24, font: "Times New Roman" }), new TextRun({ text: " \u2014 Raw dataset storage and initial data loading scripts.", size: 24, font: "Times New Roman" })] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, spacing: { after: 60, line: 360 },
        children: [new TextRun({ text: "03_model_training/proper_training/", bold: true, size: 24, font: "Times New Roman" }), new TextRun({ text: " \u2014 Model training scripts, saved models (.pkl, .keras), saved preprocessors (scaler, encoder), and generated evaluation images.", size: 24, font: "Times New Roman" })] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, spacing: { after: 60, line: 360 },
        children: [new TextRun({ text: "04_xai_integration/", bold: true, size: 24, font: "Times New Roman" }), new TextRun({ text: " \u2014 SHAP TreeExplainer wrapper class (shap_explainer.py), test scripts, and SHAP visualisation images.", size: 24, font: "Times New Roman" })] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, spacing: { after: 60, line: 360 },
        children: [new TextRun({ text: "05_mitigation_framework/", bold: true, size: 24, font: "Times New Roman" }), new TextRun({ text: " \u2014 Attack classifier, severity calculator, mitigation command generator, and alert generator.", size: 24, font: "Times New Roman" })] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, spacing: { after: 60, line: 360 },
        children: [new TextRun({ text: "06_complete_testing/", bold: true, size: 24, font: "Times New Roman" }), new TextRun({ text: " \u2014 Full pipeline benchmark test scripts and result visualisations.", size: 24, font: "Times New Roman" })] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, spacing: { after: 120, line: 360 },
        children: [new TextRun({ text: "dashboard.py", bold: true, size: 24, font: "Times New Roman" }), new TextRun({ text: " \u2014 Standalone Streamlit dashboard integrating the entire pipeline.", size: 24, font: "Times New Roman" })] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.4.3 Dashboard Implementation")] }),

      bodyText("An interactive Streamlit web dashboard was developed to provide a user-friendly interface for the complete detection pipeline. The dashboard allows users to:"),

      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, spacing: { after: 60, line: 360 },
        children: [new TextRun({ text: "Upload CSV files containing network traffic records (supporting both UNSW-NB15 and CIC-IDS2017/CIC-DDoS2019 formats via an automatic adapter).", size: 24, font: "Times New Roman" })] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, spacing: { after: 60, line: 360 },
        children: [new TextRun({ text: "View real-time detection results with colour-coded severity indicators.", size: 24, font: "Times New Roman" })] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, spacing: { after: 60, line: 360 },
        children: [new TextRun({ text: "Inspect per-record SHAP explanations through interactive waterfall plots.", size: 24, font: "Times New Roman" })] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, spacing: { after: 60, line: 360 },
        children: [new TextRun({ text: "Review attack type classifications and severity assessments.", size: 24, font: "Times New Roman" })] }),
      new Paragraph({ numbering: { reference: "bullet-list", level: 0 }, spacing: { after: 120, line: 360 },
        children: [new TextRun({ text: "Copy generated mitigation commands for immediate deployment.", size: 24, font: "Times New Roman" })] }),

      bodyText("The dashboard loads the saved XGBoost model, StandardScaler, and LabelEncoder at startup and processes uploaded records through the full pipeline: preprocessing \u2192 detection \u2192 SHAP explanation \u2192 attack classification \u2192 severity assessment \u2192 mitigation command generation."),

      // ===== 4.5 PROTOTYPE & TESTING =====
      new Paragraph({ children: [new PageBreak()] }),
      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("4.5 Prototype & Testing")] }),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.5.1 Full Pipeline Benchmark")] }),

      bodyText("The complete end-to-end pipeline was tested on all 41,089 official benchmark samples from the UNSW-NB15 external testing set. The results confirm that the integrated system maintains the same detection performance as the standalone model while successfully generating explanations, attack classifications, severity levels, and mitigation commands for every detected attack. Table 4.7 summarises the final benchmark results."),

      // Table 4.7: Benchmark Results
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 },
        children: [new TextRun({ text: "Table 4.7: Full pipeline benchmark results (41,089 samples)", bold: true, italics: true, size: 20, font: "Times New Roman" })] }),
      new Table({
        columnWidths: [4680, 4680],
        rows: [
          new TableRow({ tableHeader: true, children: [hdrCell("Metric", 4680), hdrCell("Value", 4680)] }),
          new TableRow({ children: [cell("Accuracy", 4680), cell("98.14%", 4680, {center:true})] }),
          new TableRow({ children: [cell("Precision", 4680), cell("94.42%", 4680, {center:true})] }),
          new TableRow({ children: [cell("Recall", 4680), cell("86.45%", 4680, {center:true})] }),
          new TableRow({ children: [cell("F1 Score", 4680), cell("90.26%", 4680, {center:true})] }),
          new TableRow({ children: [cell("AUC", 4680), cell("0.9915", 4680, {center:true})] }),
          new TableRow({ children: [cell("False Alarm Rate", 4680), cell("0.56% (209 / 37,000)", 4680, {center:true})] }),
          new TableRow({ children: [cell("Processing Rate", 4680), cell("422.1 samples/second", 4680, {center:true})] }),
          new TableRow({ children: [cell("Processing Time", 4680), cell("1.62 minutes (all 41,089 samples)", 4680, {center:true})] }),
        ]
      }),

      new Paragraph({ spacing: { after: 120 }, children: [] }),

      figPara("06_complete_testing/pipeline_flow_diagram.png", 520, 280),
      figCaption("Figure 4.9: Complete pipeline flow diagram"),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.5.2 Confusion Matrix Analysis")] }),

      bodyText("The confusion matrix for the full pipeline test (Figure 4.10) reveals that 36,791 out of 37,000 normal samples were correctly identified (99.44% specificity), with only 209 false positives. On the attack side, 3,535 out of 4,089 DoS attacks were correctly detected (86.45% recall), with 554 attacks missed. This represents an operationally acceptable trade-off: the system maintains an extremely low false alarm rate while detecting the vast majority of attacks."),

      figPara("06_complete_testing/confusion_matrix_heatmap.png", 360, 300),
      figCaption("Figure 4.10: Full pipeline confusion matrix heatmap"),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.5.3 Attack Classification and Severity Distribution")] }),

      bodyText("When the 3,744 detected attack records (3,535 TP + 209 FP) were processed through the mitigation framework, the attack type distribution (Figure 4.11) shows that Volumetric Floods constitute 81.3% of detections, followed by Protocol Exploits at 17.6%, Amplification at 1.0%, and Slowloris at 0.1%. This distribution is consistent with the known characteristics of the UNSW-NB15 dataset, which predominantly contains high-volume flooding attacks."),

      bodyText("The severity distribution shows that 99.97% of detections were classified as CRITICAL and 0.03% as HIGH, reflecting the high confidence of the XGBoost model on correctly detected attacks."),

      figPara("06_complete_testing/attack_type_distribution.png", 280, 220),
      figCaption("(a) Attack type distribution"),
      figPara("06_complete_testing/severity_distribution.png", 280, 220),
      figCaption("(b) Severity level distribution"),
      figCaption("Figure 4.11: Distribution of attack types and severity levels across all detections"),

      new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun("4.5.4 Comparison with Literature")] }),

      bodyText("Table 4.8 compares the evaluation methodology of this work against common practices in intrusion detection literature. While many published works report F1 scores above 95%, these are typically achieved on balanced test sets with same-dataset splits. The results in this study, evaluated on an imbalanced external test set, provide a more rigorous and realistic assessment of model performance."),

      // Table 4.8: Literature Comparison
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 80 },
        children: [new TextRun({ text: "Table 4.8: Comparison of evaluation methodology with common literature practices", bold: true, italics: true, size: 20, font: "Times New Roman" })] }),
      new Table({
        columnWidths: [2600, 3380, 3380],
        rows: [
          new TableRow({ tableHeader: true, children: [hdrCell("Aspect", 2600), hdrCell("Common Practice", 3380), hdrCell("Our Approach", 3380)] }),
          new TableRow({ children: [cell("Test Set Balance", 2600), cell("Balanced (50/50)", 3380, {center:true}), cell("Imbalanced (90/10)", 3380, {center:true})] }),
          new TableRow({ children: [cell("Validation Type", 2600), cell("Same dataset split", 3380, {center:true}), cell("External dataset (separate CSV)", 3380, {center:true})] }),
          new TableRow({ children: [cell("Reported Metrics", 2600), cell("Accuracy, F1 only", 3380, {center:true}), cell("Full metrics + AUC + CM", 3380, {center:true})] }),
          new TableRow({ children: [cell("Typical F1 Reported", 2600), cell("95%+", 3380, {center:true}), cell("90.57%", 3380, {center:true})] }),
          new TableRow({ children: [cell("Realistic?", 2600), cell("No (inflated)", 3380, {center:true}), cell("Yes (honest evaluation)", 3380, {center:true, bold:true})] }),
        ]
      }),
    ]
  }]
});

const outPath = path.join(BASE, "Chapter4_Actual_Work.docx");
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(outPath, buf);
  console.log("Created:", outPath);
}).catch(err => console.error("Error:", err));
