# Classifier selection

## DESlib

`pip install git+https://github.com/scikit-learn-contrib/DESlib`

## DESlib doc

- [https://deslib.readthedocs.io/en/latest/](https://deslib.readthedocs.io/en/latest/)

## Files

- `StreamGenerator.py` — stream generator,
- `Dumb.py`            — basic data stream classifier,
- `DESlibStream.py`    — data stream classifier employing DES methods,
- `MDE.py`             — Minority Driven Ensemble classifier,
- `TestAndTrain.py`    — learing module,
- `experiment_1.py`    — Hyperparameters overview,
- `experiment_2.py`    — MDE and DES w/ oversampling,
- `experiment_3.py`    — MDE and DES w/o oversampling.

<!--
## Uwagi
-->
<!--
1. Scorer definiowany jest bezpośrednio w klasyfikatorze, tu dla przykładu jako f1_score.
2. Metoda `fit()` klasyfikatora nie wykonuje się w przetwarzaniu ani razu, od początku wykorzystując do uczenia metodę `partial_fit()`. Pozostaje w implementacji tylko dla zaspokojenia minimum niezbędnego, aby scikit-learn traktował go jako poprawny estymator.
3. Do przetwarzania wybrano osiemnaście przykładowych strumieni syntetycznych zawierających problemy binarne o różnej trudności i różnym rodzaju dryfu koncepcji.
4. Każdy ze strumieni ma po sto tysięcy obiektów.
-->
<!--
## Przykładowy przebieg
-->
<!--
![](foo.png)
-->
