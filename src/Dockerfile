# ==========================================
# Stage 1: Build Stage
# ==========================================
FROM gcc:latest AS builder
WORKDIR /build

COPY data.cpp data.h net.cpp net.h main.cpp ./
RUN g++ -std=c++17 -O3 main.cpp net.cpp data.cpp -o mnist_net

# ==========================================
# Stage 2: Runtime Stage
# ==========================================
FROM ubuntu:24.04
WORKDIR /app

COPY --from=builder /build/mnist_net .
COPY data/ ./data/

ENTRYPOINT ["./mnist_net"]
CMD ["relu", "relu", "42"]
