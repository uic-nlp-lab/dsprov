## makefile automates the build and deployment for python projects


## Build
#
PROJ_TYPE =		python
PROJ_MODULES =		git python-resources python-cli python-doc python-doc-deploy markdown
INFO_TARGETS +=		appinfo
PY_DEP_POST_DEPS +=	modeldeps
PIP_ARGS +=		--use-deprecated=legacy-resolver
ADD_CLEAN_ALL +=	$(DIST_DIR)
CLEAN_ALL_DEPS +=	cleanpaper

## Project
#
ENTRY =			./harness.py
DIST_DIR = 		dist


include ./zenbuild/main.mk

.PHONY:			appinfo
appinfo:
			@echo "app-resources-dir: $(RESOURCES_DIR)"

.PHONY:			modeldeps
modeldeps:
			$(PIP_BIN) install $(PIP_ARGS) -r src/python/requirements-model.txt --no-deps


.PHONY:			results
results:
			rm -fr results
			mkdir -p results/csv
			$(ENTRY) excel --out results/discharge-summary-match.xlsx
			$(ENTRY) csv --out results/csv --yaml results/config
			./src/bin/opthyper.py dumpagg

.PHONY:			extract
extract:
			rm -fr $(DIST_DIR)
			mkdir -p $(DIST_DIR)
			$(ENTRY) extract --section --text --annotator
			mv annotations.json dist/dsprov-annotations-all.json
			$(ENTRY) extract --section
			mv annotations.json dist/dsprov-annotations.json

.PHONY:			cleanpaper
cleanpaper:
			make -C paper cleanall
