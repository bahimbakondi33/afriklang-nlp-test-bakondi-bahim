# afriklang-nlp-test-bakondi-bahim
Classification NLP de commentaires citoyens sur les services publics (FR + éwé/mina) – Test technique Afriklang / TAISS 2026
# Test technique NLP — Afriklang / Togo AI Lab (TAISS 2026)

Classification de 150 commentaires citoyens sur les services publics en 3 classes : `Satisfaction`, `Insatisfaction`, `Suggestion`.

**Approche et choix**
- **Exploration** : classes parfaitement équilibrées (50/50/50) ; textes très courts (≈ 9 à 10 mots) ; 70 % du vocabulaire n'apparaît qu'une seule fois.
- **Nettoyage** : minuscules, suppression de la ponctuation, des chiffres et des accents, stopwords français (liste maison qui **conserve les négations** et les intensificateurs).
- **Éwé/mina** : seulement 3 textes concernés. Je les repère, je traduis les 2 mots dont le sens est sûr (`akpe` → merci, `nyuie` → bien) et je conserve les autres tokens tels quels.
- **Vectorisation** : TF-IDF sur n-grammes de caractères (2–5), choisi après comparaison en CV avec CountVectorizer et TF-IDF mots (robuste aux variantes, aux fautes et aux mots éwé/mina).
- **Modèles** : régression logistique, SVM linéaire, Naive Bayes, Random Forest ; split 80/20 stratifié, `random_state=42`.
- **Évaluation** : accuracy, F1 macro, matrices de confusion, plus une CV 5×3 car le test ne compte que 30 textes.
- **Résultats** : les 4 modèles font 0,83 d'accuracy sur le test ; en CV, la régression logistique est en tête (F1 macro ≈ 0,73 ± 0,09), les écarts entre modèles sont non significatifs.
- **Erreurs** : la principale confusion oppose Satisfaction et Insatisfaction (négations composées, mots rares, mots de sujet partagés).
- **Limites et pistes** : peu de données, sac de n-grammes aveugle à la syntaxe ; pistes : fine-tuning multilingue (CamemBERT/XLM-R), plus de données et corpus éwé/mina annoté, variables linguistiques (négation, conditionnel).

## Reproduire
```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_analyse_nlp.ipynb   # exécuter toutes les cellules
```
Testé avec Python 3.12, scikit-learn 1.8, pandas 3.0. Les figures et le tableau de synthèse sont enregistrés dans `results/`.

## Structure
`data/` jeu de données (CSV) · `notebooks/` notebook d'analyse · `results/` figures et synthèse · `requirements.txt`
