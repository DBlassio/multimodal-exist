# inference/api/Dockerfile
#   cd inference/api
#   docker build -t multimodal-exist-api .

FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY main.py data_loader.py ./

COPY static ./static

# Lambda will invoke this object: inside main.py, the variable `handler`
CMD ["main.handler"]