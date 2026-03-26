BIN_OUTPUT_PATH = bin
MODULE_BINARY = $(BIN_OUTPUT_PATH)/icp-slam
COMMON_LDFLAGS = -s -w

build:
	mkdir -p $(BIN_OUTPUT_PATH)
	rm -f $(MODULE_BINARY)
	go build -ldflags="$(COMMON_LDFLAGS)" -o $(MODULE_BINARY) cmd/module/main.go

module: build
	rm -f $(BIN_OUTPUT_PATH)/module.tar.gz
	tar czf $(BIN_OUTPUT_PATH)/module.tar.gz -C $(BIN_OUTPUT_PATH) icp-slam -C .. meta.json

test:
	go test -race ./...

lint:
	go mod tidy
	go vet ./...

clean:
	rm -rf $(BIN_OUTPUT_PATH)

.PHONY: build module test lint clean
