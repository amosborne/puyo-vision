# Generate python package dependencies without Linux bug.
freeze:
	pip freeze | grep -v "pkg-resources" > requirements.txt
