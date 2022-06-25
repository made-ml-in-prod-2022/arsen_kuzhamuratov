FROM python:3.7
COPY requirements.txt /tmp/
RUN pip install --no-cache -r /tmp/requirements.txt && rm /tmp/requirements.txt
COPY . .
RUN pip install --no-cache -e .
ENV CONFIG_PATH='./configs/grad_boosting.yaml'
CMD ["python", "./online_inference/app.py"]