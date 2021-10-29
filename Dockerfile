FROM ubuntu:latest

WORKDIR /usr/src/app/

RUN apt update && apt install -y python3-pip
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY prod /usr/src/app/prod/
COPY train.py /usr/src/app/
COPY predict.py /usr/src/app/

RUN python3 train.py -d dummy.csv -mp model

CMD [ "python3", "train.py", "-d", "dummy.csv", "-mp", "model", "-o", "res.csv" ]