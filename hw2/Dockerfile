FROM python:3.7
COPY . .
RUN pip install -r requirements.txt
RUN pip install -e .
ENV CONFIG_PATH='./configs/grad_boosting.yaml'
CMD ["python", "./online_inference/app.py"]