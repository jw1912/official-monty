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
	cargo rustc --release --package monty --bin monty -- --emit link=$(NAME)
