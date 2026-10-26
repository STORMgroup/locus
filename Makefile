CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -Wpedantic

ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
BUILD := $(ROOT)/build
LCPTOOLS_SRC := $(ROOT)/third_party/lcptools
LCPTOOLS_PREFIX := $(BUILD)/lcptools
LCPTOOLS_LIB := $(LCPTOOLS_PREFIX)/lib/liblcptools.so
SPLIT_FASTA := $(BUILD)/split_fasta
# PBSIM3 (Ono et al. 2022), only for the simulated nanopore reads of the tokenizer study (benchmarks/noisy_reads)
PBSIM3_COMMIT := 35ce0debb8b9df8a45ac9bb85420686776e9e721
PBSIM3 := $(BUILD)/pbsim3/src/pbsim

.PHONY: all lcptools tools pbsim3 clean

all: lcptools tools

# LD_RUN_PATH (set, for example, by CUDA environment modules) would be written into the binaries as a library search path;
# it is cleared so the build is the same in every environment.
NO_RUN_PATH := env -u LD_RUN_PATH

# LCP library used by every model in the paper: label and position width 32 bit, dictionary encoding on.
lcptools: $(LCPTOOLS_LIB)

$(LCPTOOLS_LIB): $(wildcard $(LCPTOOLS_SRC)/*.c) $(wildcard $(LCPTOOLS_SRC)/*.h) $(LCPTOOLS_SRC)/config.h.in
	mkdir -p $(LCPTOOLS_PREFIX)
	$(NO_RUN_PATH) $(MAKE) -C $(LCPTOOLS_SRC) install PREFIX=$(LCPTOOLS_PREFIX) LABEL=32 POS=32 DCT=1

tools: $(SPLIT_FASTA)

$(SPLIT_FASTA): $(ROOT)/tools/split_fasta.c | $(BUILD)
	$(NO_RUN_PATH) $(CC) -I$(ROOT)/third_party/htslib $(CFLAGS) $< -lz -o $@

pbsim3: $(PBSIM3)

$(PBSIM3): | $(BUILD)
	rm -rf $(BUILD)/pbsim3
	git clone --quiet https://github.com/yukiteruono/pbsim3.git $(BUILD)/pbsim3
	git -C $(BUILD)/pbsim3 checkout --quiet $(PBSIM3_COMMIT)
	cd $(BUILD)/pbsim3 && ./configure > configure.log 2>&1 && $(MAKE) > make.log 2>&1

$(BUILD):
	mkdir -p $@

clean:
	$(MAKE) -C $(LCPTOOLS_SRC) clean || true
	rm -rf $(BUILD)
