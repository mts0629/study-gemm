.PHONY: all format build clean

all: format build

build:
	@cmake -S . -B ./build
	@cmake --build ./build

format:
	@clang-format-13 -i ./include/* ./src/* ./test/*

clean:
	@rm -rf ./build
