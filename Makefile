EXE = monty

ifeq ($(OS),Windows_NT)
	NAME := $(EXE).exe
else
	NAME := $(EXE)
endif

INVOKE := RUSTFLAGS="-Ctarget-cpu=native" cargo +stable rustc --release
LINK := -- --emit link=$(NAME)

default:
	$(INVOKE) --bin monty --features=embed $(LINK)
	
raw:
	$(INVOKE) --bin monty --features=embed,raw $(LINK)

montytest:
	$(INVOKE) --bin monty --features=uci-minimal,tunable $(LINK)

noembed:
	$(INVOKE) --bin monty $(LINK)

gen:
	$(INVOKE) --package datagen --bin datagen $(LINK)