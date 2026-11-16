# Catching Forged PDFs by Their Handwriting

**Adversarially Robust Detection and Localization of Tampered PDF Documents via Compression-Consistency Analysis**

## The problem

Someone opens a genuine PDF (a salary slip, a grade sheet, an invoice), changes one number, and saves it. The file opens normally, the new digit is drawn by the same font in the same place, and nothing looks wrong. The person checking it usually has **only the file**: no original to compare with and no digital signature.

Today's checks don't catch this:

| Approach | Why it fails on this kind of edit |
|---|---|
| Image forensics (e.g. TruFor, CVPR 2023) | A PDF page is drawn fresh from its font, so there are no pixel traces |
| Digital signatures / embedded hashes | Need the issuer to sign in advance; most PDFs are unsigned |
| Counting saves (`%%EOF` markers) | Honest web-optimised files have two markers; a forger's normal save has one |
| Structure-based detectors (PDFrate, Hidost, coding style) | They detect that a tool *touched* the file, but honest re-saves are touched too |

**Our question:** given only a suspicious PDF, was it edited after it was made, and **on which page**?

## The idea: compression "handwriting"

Each page's text in a PDF is stored zipped (zlib/Flate). Every program zips in its own way: header bytes, whether standard zlib can recreate the exact bytes, checksum, how the stream ends, extra bytes, packing ratio. For example, Adobe-made pages start with `48 89` in 96–100% of cases, while the zlib inside Python tools writes `78 9C` by default.

A genuine file is written once by one program, so **every page has the same handwriting**. To change one digit, an editing tool must unzip that page, edit it, and zip it again with its own zlib, while the other pages are copied byte for byte. The edited page becomes **the odd one out**. That tells us:

- that the file was edited,
- **which page** was changed,
- without needing the original.

```
Genuine file:  P1 48 89 | P2 48 89 | P3 48 89 | P4 48 89 | P5 48 89
Edited file:   P1 48 89 | P2 48 89 | P3 78 9C | P4 48 89 | P5 48 89
                                     ^ re-zipped by the editing tool
```

**Known blind spots:** re-zipping every page ("laundering"), content only added on top (a stamp or a text box), and a forger who copies the handwriting (studied below).

## Method

```
Suspicious PDF ──► read every object, unzip page text
                    ├─► 61 clues ─────────────► LightGBM detector ──► edited? (score > 0.531)
                    └─► object graph ─────────► GNN page-finder  ──► which page?
```

### 61 clues in 5 groups

| Group | # | What it looks at |
|---|---|---|
| **Compression consistency (core)** | 17 | How each page's text is zipped, compared across all pages: headers, zip settings, reproducibility, signatures, odd-one-out streams |
| File style | 27 | How the file is written: line endings, spacing, object order, xref type, number of saves |
| Last update | 5 | What the newest appended save changed |
| Page resources | 8 | Annotations, text boxes, images, fonts, streams per page |
| Metadata | 5 | Producer, dates, tool names (weak, easy to fake) |

`tool_word_in_file` is counted in both Metadata and File style (62 slots, 61 distinct clues). Removing the 17 compression clues drops the evidence-bearing AUC from 0.998 to 0.874; removing any other group changes almost nothing.

### Detector: LightGBM

- 300 trees, learning rate 0.05, ≤ 15 leaves, min 10 files per leaf, subsample 0.8, colsample 0.8, seed 42
- One row = one PDF; trained on 34,785 files from 2,319 documents
- Threshold **0.531**: the 95th percentile of out-of-fold scores of honest training files (5-fold, grouped by document), about 5% false alarms

### Page-finder: graph neural network (CGNN_DOC)

- Nodes = PDF objects, edges = references (12 types, plus reverse edges)
- 65 features per node: 43 about the object itself (type, size, filter, zlib header and setting, reproducibility, checksum, ending, extra bytes, packing ratio, entropy, …) and 22 comparing a page's text stream with the others
- 3 GINEConv layers (hidden 64, dropout 0.2) with edge-type embeddings, plus the 61 document clues through a small MLP
- **Weak supervision:** trained only on "edited / not edited", never told the page; the highest-scoring page-text node is the predicted page
- AdamW (lr 2e-3, weight decay 1e-4), batch 32, ReduceLROnPlateau, early stopping (patience 10, max 60 epochs), seeds 0/1/2

## Dataset

Built from **GovDocs1** (Garfinkel et al., 2009), zips 000–019:

```
5,029 PDFs ─► 3,467 clean ─► 2,914 originals (5 producer programs) ─► 2,722 editable documents ─► × 15 versions = 40,830 files
```

"Clean" means: the file opens, is not encrypted, was never edited before, has fonts, has no signature, and has at most 50 pages.

**15 versions per document.** Every forgery has an **honest twin**: the same tool and the same kind of save, without the edit. Without twins, any method that just notices "a tool touched this file" looks perfect.

| Honest | | Forged | Twin | Odd page possible? |
|---|---|---|---|---|
| G0 | untouched original | | | |
| G1 | open and re-save | F1 one digit · F4 one amount · F5 white-box cover-up | G1 | yes |
| G2 | re-zip every page | F2 digit + re-zip every page | G2 | no |
| G3 | sticky note, appended save | F3 digit, appended save | G3 | yes |
| G4 | web-optimise | F6 digit + web-optimise | G4 | yes |
| G5 | "RECEIVED" stamp | F7 fake "PAID" stamp | G5 | no |
| G6 | "Approved" text box | F8 fake "TOTAL DUE" text box | G6 | no |

- Edits made with **pikepdf, PyMuPDF and pypdf** (three different engines: qpdf, MuPDF, pure Python), picked per document by a fixed seed (`crc32(name) + 2026`)
- **Split by document**, never by file: dev if `crc32(file name) % 100 < 15` → 2,319 train documents (34,785 files) / 403 dev documents (6,045 files)

## Results (development set)

| Result | Value |
|---|---|
| AUC on evidence-bearing files (an odd page can appear; 3,627 files) | **0.998** (95% CI 0.997–0.999) |
| Forgeries caught at 1% false alarms (evidence-bearing) | **96.5%** |
| Untouched originals flagged | **0 of 403** |
| AUC over all 15 versions (includes the blind spots) | 0.908 (95% CI 0.905–0.911) |
| Twin AUC: one digit / one amount / digit + web-optimise | 0.996 / 0.996 / 0.99 |
| Leave-one-tool-out (re-save forgeries): pikepdf / PyMuPDF / pypdf | 0.993 / 0.999 / 0.99 |
| Right page named by the GNN (F1 / F4 / F5 / F6) | 95.5 / 95.8 / 100.0 / 89.2 % |

### Comparison with published methods

Common development set: 5,955 files from 397 documents, scored by every method. All baselines were retrained on our forged/honest labels.

| Method | AUC (evidence) | Twin F1 | Twin F4 | Twin F6 |
|---|---|---|---|---|
| `%%EOF > 1` rule | 0.377 | 0.500 | 0.500 | 0.500 |
| TruFor (CVPR 2023, official weights, told the edited page) | 0.500 | 0.500 | 0.500 | 0.500 |
| Hidost (EURASIP JIS 2016, authors' tools) | 0.693 | 0.503 | 0.503 | 0.501 |
| Coding style (our approximation of Adhatarao & Lauradoux, 2021) | 0.850 | 0.500 | 0.500 | 0.500 |
| PDFrate (ACSAC 2012, Mimicus extractor) | 0.867 | 0.504 | 0.509 | 0.504 |
| **Ours** | **0.998** | **0.996** | **0.996** | **0.989** |

On the twin test, every published method is at a coin toss: a one-digit edit changes no structure, counts or style, only how one page is zipped.

## Robustness: attacking our own detector

1,003 documents attacked (403 dev + 600 train); results on the 403 dev documents, unchanged detector, threshold 0.531.

| Attack | Fools the detector |
|---|---|
| A0 plain digit edit | 0.5% |
| A2a simple re-zip | 0.5% |
| **A2b clever copy** (re-zip the edited page with the original page's header, setting, ending, checksum, extra bytes and ratio) | **84.6%** |
| A3 hide metadata traces | 0.5% |
| A4 invisible decoy streams | 0.2% |
| A5c A2b + A4 + A3 | 84.4% |

Standard defenses barely help: attack training (D1), robust features (D2) and monotonic constraints (D3) together catch 31.0% of A2b.

### Defense D4: could ordinary zlib have made this exact page?

Each page-text stream is unzipped and re-zipped with **every** standard zlib setting matching its own window size. Pages from Adobe's own compressor (Distiller, PDF Library) are almost never reproducible (0–8%); the forger's copied page was made with zlib, so it is, and it becomes the odd one out again.

| | No defense | **FINAL** (D4 + attack training + honest re-saves) |
|---|---|---|
| A2b caught | 15.4% | **78.7%** (95% CI 74.4–82.6) |
| A5c caught | 15.6% | 99.5% |
| Evidence-bearing AUC | 0.9981 | 0.9978 |
| Untouched originals flagged | 0.0% | 0.0% |
| Honest pikepdf re-saves flagged | 12.4% | 25.6% |

D4 helps only documents from programs that use their own compressor (about 4 in 5 of ours). PDFWriter and Corel pages are mostly zlib-reproducible already.

## Repository structure

The code runs as Kaggle notebooks. Each cell starts with `# CELL N — purpose`.

```
notebooks/
├── BTP-full            # CELL 20–22  dataset build: 15 versions, 61 clues, object graphs
├── BTP-train           # CELL 23, 24, 26  LightGBM, GNN setup, final dev comparison
├── BTP-gnn-GIN         # CELL 25 (KIND="GIN")       train 3 seeds
├── BTP-gnn-CGNN        # CELL 25 (KIND="CGNN")      train 3 seeds
├── BTP-gnn-DOC         # CELL 25 (KIND="CGNN_DOC")  train 3 seeds
├── BTP-attack          # CELL 27–33c  attacks A0–A5c, defenses D1–D4, FINAL defense
├── BTP-baselines       # CELL 41, 44  baseline tool setup + feature extraction
├── BTP-baseline-trufor # CELL 46–47  TruFor (official code + weights)
└── BTP-baselines-2     # CELL 45, 48  baseline evaluation + final comparison table
```

> Adjust this tree if your folder names differ.

## Reproducing

1. **Data:** attach GovDocs1 zips 000–019 and the clean-pool selection (`clean_pool.csv`) as Kaggle inputs.
2. **Build the dataset:** run `BTP-full` (CELL 20 → 22). Output: `full/full_features.csv`, `full/full_graphs.pkl.gz`.
3. **Detector:** run `BTP-train` CELL 23.
4. **GNN:** run CELL 24 setup, then CELL 25 in each of the three GNN notebooks, then CELL 26 for the comparison.
5. **Attacks and defenses:** run `BTP-attack` CELL 27 → 33c.
6. **Baselines:** `BTP-baselines` (41, 44), `BTP-baseline-trufor` (46, 47), then `BTP-baselines-2` (45, 48).

Notebooks pass outputs to each other through Kaggle **Add Input → Your Work → Notebooks**.

**Main dependencies:** `pikepdf`, `pymupdf`, `pypdf`, `pebble`, `lightgbm`, `scikit-learn`, `torch`, `torch_geometric`. Baselines also need Python 2.7 (Mimicus/PDFrate), the Hidost C++ tools (Poppler 0.57) and the official TruFor repository and weights.

## Limitations

- No odd page means no detection: laundering, stamps and text boxes (twin AUC ≈ 0.5)
- We localize the **page**, not the exact word
- D4 cannot help when genuine pages are already zlib-made (PDFWriter, Corel)
- Forgeries were made with three Python tools on older government files; real editors (Adobe Acrobat, Word, online editors) are not tested yet
- GNN weights were not saved (only predictions), so the page-finder has not been tested under attack yet
- Baseline features came from files regenerated with newer library versions (same seeds and edits)
- All numbers are development-set numbers

## Next steps

- [ ] Quick checks: model without `tool_word_in_file`, random-page localization baseline, Corel false alarms
- [ ] Unseen attacks: a different compressor (A6), trial-and-error against the detector (A5)
- [ ] Retrain the GNN with saved weights and test it under attack
- [ ] Real-editor study: online editors, Word, LibreOffice
- [ ] PDFObj2Vec baseline (ACM CCS 2025)
- [ ] Freeze the system, build a fresh test set from new GovDocs zips, evaluate once
- [ ] Paper (candidate venues: DFRWS / FSI: Digital Investigation, ARES, IH&MMSec)

## References

1. C. Smutz, A. Stavrou. Malicious PDF detection using metadata and structural features. ACSAC 2012.
2. N. Šrndić, P. Laskov. Practical evasion of a learning-based classifier: a case study. IEEE S&P 2014.
3. N. Šrndić, P. Laskov. Hidost: a static machine-learning-based detector of malicious files. EURASIP J. Information Security, 2016.
4. F. Guillaro et al. TruFor: leveraging all-round clues for trustworthy image forgery detection and localization. CVPR 2023.
5. S. Adhatarao, C. Lauradoux. Robust PDF files forensics using coding style. arXiv:2103.02702, 2021.
6. G. Grobler, S. Makura, H. Venter. A technique for the detection of PDF tampering or forgery. arXiv:2507.00827, 2025.
7. S. Liu et al. PDFObj2Vec. ACM CCS 2025.
8. L. Grinsztajn, E. Oyallon, G. Varoquaux. Why do tree-based models still outperform deep learning on typical tabular data? NeurIPS 2022.
9. S. Garfinkel, P. Farrell, V. Roussev, G. Dinolt. Bringing science to digital forensics with standardized forensic corpora. Digital Investigation, 2009.
10. P. Deutsch, J-L. Gailly. ZLIB compressed data format specification version 3.3. RFC 1950, 1996.

## Acknowledgements

We thank our guide, Dr. Gunjan Chugh, and the Department of Computer Science & Engineering, Delhi Technological University.
