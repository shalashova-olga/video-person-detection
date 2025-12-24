# -------------------------------
# 1️⃣ Базовый образ с Python
# -------------------------------
    FROM python:3.11-slim

    # Локаль и прочее
    ENV LANG=C.UTF-8
    ENV LC_ALL=C.UTF-8
    ENV SPARK_LOCAL_HOSTNAME=localhost
    
    # Airflow директория
    ENV AIRFLOW_HOME=/usr/local/airflow
    ENV PROJECT_DIR=/usr/local/airflow/bigdata_project
    
    # -------------------------------
    # 2️⃣ Системные зависимости
    # -------------------------------
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        default-jdk-headless \
        libpq-dev \
        curl \
        graphviz \
        && rm -rf /var/lib/apt/lists/*
    
    # -------------------------------
    # 3️⃣ Рабочая директория
    # -------------------------------
    WORKDIR $PROJECT_DIR
    
    # -------------------------------
    # 4️⃣ Python-зависимости
    # -------------------------------
    COPY requirements.txt ./requirements.txt
    RUN pip install --no-cache-dir --upgrade pip \
        && pip install --no-cache-dir -r requirements.txt \
        && pip install --no-cache-dir apache-airflow==2.9.3
    
    # -------------------------------
    # 5️⃣ Копируем проект
    # -------------------------------
    COPY . .
    
    # -------------------------------
    # 6️⃣ Порты
    # -------------------------------
    EXPOSE 8080 8501
    
    # -------------------------------
    # 7️⃣ Точка входа
    # -------------------------------
    ENTRYPOINT ["bash", "-c"]
    CMD ["airflow standalone"]
    