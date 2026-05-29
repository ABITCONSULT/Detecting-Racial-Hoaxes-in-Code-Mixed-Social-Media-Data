# EM-26@LT-EDI 2025: Detecting Racial Hoaxes in Code-Mixed Social Media Data

This repository contains the official system description code and experimental pipeline implemented by team **EM-26** for the **LT-EDI Shared Task at LDK 2025** (organized by DravidianLangTech).

The project introduces an optimized transformer classification framework built to parse user-generated social media text for **racial hoaxes** and aggressive behavior when written in highly unstructured, bilingual **code-mixed** language profiles.

---

##  Abstract

Social media platforms and user-generated content, such as tweets, comments, and blog posts often contain offensive language, including racial hate speech, personal attacks, and sexual harassment. Detecting such inappropriate language is essential to ensure user safety and to prevent the spread of hateful behavior and online aggression. Approaches base on conventional machine learning and deep learning have shown robust results for high-resource languages like English and find it hard to deal with code-mixed text, which is common in bilingual communication. We participated in the shared task "LT-EDI@LDK 2025" organized by DravidianLangTech, applying the BERT-base multilingual cased model and achieving an F1 score of $0.63$. These results demonstrate how our model effectively processes and interprets the unique linguistic features of code-mixed content. The source code is available on GitHub.

---

##  Handling the Code-Mixed Linguistic Challenge

Traditional monolingual NLP pipelines struggle with code-mixed text because switching mid-sentence introduces immense out-of-vocabulary (OOV) noise and broken syntax.

Our pipeline approaches this challenge through targeted architectural choices:

* **Tokenization Preservation:** Employs the cased **Multilingual BERT** framework (`bert-base-multilingual-cased`) to preserve case markers, which often provide strong stylistic signals in aggressive social media posts.
* **Cross-Lingual Sub-Word Tracking:** The shared WordPiece vocabulary space in mBERT enables the model to bridge morphological shifts, mapping common slang roots and mixed scripts into reliable contextual embeddings without requiring language-isolated preprocessing pipelines.

---

##  Shared Task Evaluation Results

When evaluated on the blind tournament test sets provided by the LT-EDI@LDK 2025 event organizers, our optimized fine-tuning configuration achieved a robust **F1-Score of 0.63**, demonstrating high stability against erratic bilingual social strings.

---

##  Getting Started

### Prerequisites

* Python 3.9+
* PyTorch
* Transformers (Hugging Face)
* Scikit-Learn
* Pandas, NumPy

### Installation

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/REPO_NAME.git
cd REPO_NAME
pip install -r requirements.txt

```

### Usage

1. **Preprocess and Clean Code-Mixed Text Feeds:**
```bash
python src/preprocess_codemixed.py --data_path ./data/lt_edi_2025

```



```
2. **Execute mBERT Fine-Tuning Optimization Loop:**
   ```bash
   python src/train_detector.py --model bert-base-multilingual-cased --epochs 4 --batch_size 16 --lr 2e-5

```

3. **Generate Tournament Predictions and F1 Analytics:**
```bash
python src/evaluate_metrics.py --preds ./results/test_predictions.json

```



##  Citation
If you implement this code-mixed safety classification loop or reference the EM-26 shared task results, please cite our official workshop paper:

```bibtex
@inproceedings{abiola-etal-2025-em26,
    title = "{EM}-26@{LT}-{EDI} 2025: Detecting Racial Hoaxes in Code-Mixed Social Media Data",
    author = "Abiola, Tolulope Olalekan and Ojo, Olumide Ebenezer and Sidorov, Grigori",
    booktitle = "Proceedings of the Fourth Workshop on Language Technology for Equality, Diversity, and Inclusion (LT-EDI 2025)",
    month = june,
    year = "2025",
    address = "Torino, Italy",
    publisher = "Association for Computational Linguistics",
    pages = "184--190"
}

```

---
