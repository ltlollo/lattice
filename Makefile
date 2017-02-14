SRC = src
OUT = bin
ARCH = AVX2
LANG = c

TEST_SRC_FILES := $(wildcard tests/*.$(LANG)) \
	$(wildcard tests/$(ARCH)/*.$(LANG))
BENCH_SRC_FILES := $(wildcard benches/*.$(LANG))
TESTS := $(basename $(TEST_SRC_FILES))
BENCHES := $(basename $(BENCH_SRC_FILES))

DEPS := $(wildcard $(SRC)/$(ARCH)/*.$(LANG))
INCLUDES := -I$(SRC)

CC_FLAGS := -pipe -march=native -Wall -Wextra -pedantic

OPT_FLAGS := $(CC_FLAGS) -s -O3 -ftree-vectorize -funroll-all-loops 		\
	-fprefetch-loop-arrays -fsched-pressure -fsched-spec-load 				\
	-fsched-spec-load-dangerous -fschedule-insns -fschedule-insns2 			\
	-minline-all-stringops -fsched-stalled-insns=8 -fsched2-use-superblocks \
	-ffunction-sections -fdata-sections

DBG_FLAGS := $(CC_FLAGS) -O0 -g -fno-omit-frame-pointer
LD_FLAGS := -Wl,--gc-sections

FMT := "\texec:\t\t%C\n\trealtime:\t%e\n\tpgfaults:\t%R\n\tmemused:\t%K"

test: $(TESTS)
	@for i in $(notdir $(TESTS)); do						\
		$(OUT)/$$i && echo [OK]: $$i || echo [FAIL]: $$i;	\
	done

bench: $(BENCHES)
	@for i in $(notdir $(BENCHES)); do						\
		echo [RUNNING]: $$i; time -f $(FMT) $(OUT)/$$i;		\
	done

tests/%:
	@echo [COMPILING]: $(CC) $@
	@$(CC) $(LD_FALGS) $(DBG_FLAGS) $(INCLUDES) $(DEPS) $@.$(LANG) \
		-o $(OUT)/$(notdir $@)

benches/%:
	@echo [COMPILING]: $(CC) $@
	@$(CC) $(LD_FLAGS) $(OPT_FLAGS) $(INCLUDES) $(DEPS) $@.$(LANG) \
		-o $(OUT)/$(notdir $@)

clean:
	@rm $(OUT)/*

.PHONY: clean test bench

