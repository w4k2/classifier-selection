# Classifier selection

## Pliki

- `ARFF.py` — prymitywny parser plików ARFF,
- `DumbDelayPool.py` — przykładowa klasa prymitywnego klasyfikatora strumieniowego,
- `experiment.py` — przykład przetwarzania z wykorzystaniem klasyfikatora *DumbDelayPool*,
- `TestAndTrain.py` — moduł uczący.
- `streams.txt` — lista wszystkich plików zawierających strumienie

## Teczki

- `streams` — strumienie w formacie ARFF:
  - prefiks `id` to dryf inkrementacyjny
  - prefiks `sd` to dryf nagły
- `figures` — przykładowe wykresy przebiegu wygenerowane przez `experiment.py`

## Uwagi

1. Scorer definiowany jest bezpośrednio w klasyfikatorze, tu dla przykładu jako f1_score.
2. Metoda `fit()` klasyfikatora nie wykonuje się w przetwarzaniu ani razu, od początku wykorzystując do uczenia metodę `partial_fit()`. Pozostaje w implementacji tylko dla zaspokojenia minimum niezbędnego, aby scikit-learn traktował go jako poprawny estymator.
3. Do przetwarzania wybrano osiemnaście przykładowych strumieni syntetycznych zawierających problemy binarne o różnej trudności i różnym rodzaju dryfu koncepcji.
