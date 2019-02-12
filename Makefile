prepare:
	pip install git+https://github.com/scikit-learn-contrib/DESlib

populate:
	cp -r figures ~/Dropbox/Aplikacje/Overleaf/Classifier\ selection\ for\ imbalanced\ data\ streams\ with\ Minority\ Driven\ Ensemble/

analyze:
	python analyze_1.py
	python analyze_2.py
