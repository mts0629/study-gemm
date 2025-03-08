.PHONY: all format build clean

all: build

format:
	@clang-format-13 -i ./include/* ./src/* ./test/*

build: format
	@cmake -S . -B ./build
	@cmake --build ./build

clean:
	@rm -rf ./build
