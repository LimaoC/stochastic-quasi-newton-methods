.PHONY: test_lr, check

run_spambase:
	python -m tests.classification.run_spambase

check:
	black sqnm && mypy sqnm
