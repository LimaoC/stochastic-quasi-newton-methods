.PHONY: check

run_spambase:
	python -m tests.run_spambase

check:
	black sqnm && mypy sqnm
