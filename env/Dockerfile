FROM nvidia/cuda:12.1.1-runtime-ubuntu20.04

SHELL ["/bin/bash", "-c"]

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pyenvのインストール
ENV PYENV_ROOT /root/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH
ENV PATH $PYENV_ROOT/shims:${PATH}
ENV PYTHON_VERSION 3.10.13

RUN git clone https://github.com/yyuu/pyenv.git ~/.pyenv


# RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv

# RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc && \
#     echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc && \
#     echo 'eval "$(pyenv init -)"' >> ~/.bashrc

# RUN apt-get update; apt-get install -y build-essential libssl-dev zlib1g-dev \
#     libbz2-dev libreadline-dev libsqlite3-dev curl \
#     libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
# RUN . ~/.bashrc
RUN pyenv install $PYTHON_VERSION
RUN pyenv global $PYTHON_VERSION

# Pythonをインストール
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3.10 \
#     python3-pip \
#     && rm -rf /var/lib/apt/lists/*

# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
# RUN python3 -m pip install --upgrade pip

# Poetryのインストール

CMD ["tail", "-f", "/dev/null"]