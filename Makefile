# Generate `pip freeze` requirements.txt using pigar, without comments.
freeze:
	pigar -i .venv dev -p .tmpreqs --without-referenced-comments
	tail -n +3 .tmpreqs > requirements.txt
	rm .tmpreqs
