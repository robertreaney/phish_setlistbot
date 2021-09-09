FROM python:3.6.14-buster
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
ENTRYPOINT ["/bin/bash"]