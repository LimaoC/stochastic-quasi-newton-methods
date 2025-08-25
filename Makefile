.PHONY: check

LOG_RUN = logfile="$$LOG_PATH/$$(date +%Y%m%d-%H%M%S).log"; \
		  echo $$COMMAND > $$logfile; \
		  $$COMMAND 2>&1 | tee -a $$logfile

spambase_decaying:
	@COMMAND="python -m tests.binary_classification.run_spambase -e 500 -b 100 500 -s decaying"; \
	LOG_PATH="outs/binary_classification/spambase/logs"; \
	$(LOG_RUN)

spambase_strong_wolfe:
	@COMMAND="python -m tests.binary_classification.run_spambase -e 500 -b 100 500 -s strong_wolfe -S sgd"; \
	LOG_PATH="outs/binary_classification/spambase/logs"; \
	$(LOG_RUN)

mushroom_decaying:
	@COMMAND="python -m tests.binary_classification.run_spambase -e 500 -b 100 500 -s decaying -d mushroom"; \
	LOG_PATH="outs/binary_classification/mushroom/logs"; \
	$(LOG_RUN)

spambase_plot:
	python -m tests.binary_classification.plots -s strong_wolfe

check:
	black sqnm && mypy sqnm
