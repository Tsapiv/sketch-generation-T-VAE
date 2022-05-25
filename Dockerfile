FROM python:3.9
COPY . sketch-project/
WORKDIR sketch-project/
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt
EXPOSE 8000

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]