.PHONY: all format config build asm clean

all: build

format:
	@clang-format -i ./include/* ./src/* ./test/*

config: format
	@cmake -S . -B ./build -DIMPL="$(IMPL)" # Specify implementation type 

build: config
	@cmake --build ./build

asm: config
	@cmake --build ./build --target=assembly

clean:
	@rm -rf ./build
