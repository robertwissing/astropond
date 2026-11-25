CC ?= cc
target ?= ChaNGa

fake_flags := -DFAKE_MHD=0 -DFAKE_BAROTROPIC=0 -DFAKE_EXTRA=0
ifneq (,$(filter --enable-mhd,$(CHANGA_EFFECTIVE_CONFIGURE_FLAGS)))
fake_flags := $(filter-out -DFAKE_MHD=0,$(fake_flags)) -DFAKE_MHD=1
endif
ifneq (,$(filter --enable-barotropic,$(CHANGA_EFFECTIVE_CONFIGURE_FLAGS)))
fake_flags := $(filter-out -DFAKE_BAROTROPIC=0,$(fake_flags)) -DFAKE_BAROTROPIC=1
endif
ifneq (,$(filter --enable-fixture-extra,$(CHANGA_EFFECTIVE_CONFIGURE_FLAGS)))
fake_flags := $(filter-out -DFAKE_EXTRA=0,$(fake_flags)) -DFAKE_EXTRA=1
endif

.PHONY: all
all:
	$(CC) $(fake_flags) fake.c -o $(target)
