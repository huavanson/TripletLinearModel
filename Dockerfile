# Sử dụng hình ảnh cơ sở Ubuntu 20.04
FROM ubuntu:20.04

# Cài đặt các gói cần thiết (Python, MLflow, và các thư viện khác)
RUN apt-get update -y && apt-get install -y python3 python3-pip
RUN pip3 install mlflow

# Tạo thư mục để lưu trữ dữ liệu MLflow (nếu cần)
WORKDIR /app
ENV MLFLOW_HOME /app/mlflow

# Xác định cổng mà MLflow sẽ lắng nghe (mặc định là 5000)
EXPOSE 5002

# CMD để chạy MLflow server khi container được khởi động
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5002"]