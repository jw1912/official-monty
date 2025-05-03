EXE = monty

ifeq ($(OS),Windows_NT)
	NAME := $(EXE).exe
	OLD := monty-$(VER).exe
	AVX2 := monty-$(VER)-avx2.exe
else
	NAME := $(EXE)
	OLD := monty-$(VER)
	AVX2 := monty-$(VER)-avx2
endif

default:
	cargo rustc --release --features=embed -- -C target-cpu=native --emit link=$(NAME)
	
raw:
	cargo rustc --release --features=embed,raw -- -C target-cpu=native --emit link=$(NAME)

montytest:
	cargo +stable rustc --release --features=uci-minimal,tunable -- -C target-cpu=native --emit link=$(NAME)

noembed:
	cargo rustc --release -- -C target-cpu=native --emit link=$(NAME)

gen:
	cargo rustc --release --features=datagen -- -C target-cpu=native --emit link=$(NAME)