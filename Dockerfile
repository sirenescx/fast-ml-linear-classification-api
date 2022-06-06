FROM python
WORKDIR /app

COPY . ./
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

EXPOSE 84
ENTRYPOINT [ "python", "main.py" ]