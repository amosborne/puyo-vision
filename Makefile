# Generate python package dependencies without Linux bug.
freeze:
	pigar -i .venv dev -p .tmpreqs --without-referenced-comments
	tail -n +3 .tmpreqs > requirements.txt
	rm .tmpreqs
