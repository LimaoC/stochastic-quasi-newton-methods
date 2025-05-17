.PHONY: test_lr, check

test_lr:
	python -m tests.logistic_regression.run

check:
	black sqnm && mypy sqnm
