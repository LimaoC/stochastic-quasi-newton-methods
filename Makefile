.PHONY: test_lr, check

run_spambase:
	python -m tests.classification.spambase_lbfgs
	python -m tests.classification.spambase_olbfgs
	python -m tests.classification.spambase_sqn_hv

check:
	black sqnm && mypy sqnm
