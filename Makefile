# OpenTSR convenience targets. Run from the repository root with the virtualenv active.
#
# Traffic sign targets (main track) are added here as they appear.
# Traffic light targets (secondary track) live in traffic_lights/Makefile and are reached with a
# "lights-" prefix, for example `make lights-s2tld` or `make lights-replicate-autoware`.
PY ?= python

.PHONY: test help

help:
	@echo "make test                 run the unit tests"
	@echo "make lights-<target>      run a target from traffic_lights/Makefile (see that file)"

test:
	$(PY) -m pytest -q tests

lights-%:
	$(MAKE) -f traffic_lights/Makefile PY="$(PY)" $*
