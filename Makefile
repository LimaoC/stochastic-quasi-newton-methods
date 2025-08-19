.PHONY: check

LOG_RUN = logfile="log/$$(date +%Y%m%d-%H%M%S).log"; \
		  echo $$COMMAND > $$logfile; \
		  $$COMMAND 2>&1 | tee -a $$logfile

spambase_decaying:
	@COMMAND="python -m tests.logistic_regression.run_spambase --save-fig -e 500 -b 100 -s decaying"; $(LOG_RUN)

spambase_strong_wolfe:
	@COMMAND="python -m tests.logistic_regression.run_spambase --save-fig -e 1000 -b 500 -s strong_wolfe"; $(LOG_RUN)

mnist:
	python -m tests.run_mnist

check:
	black sqnm && mypy sqnm
