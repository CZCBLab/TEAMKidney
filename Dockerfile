# syntax=docker/dockerfile:1.7-labs
# ^ the COPY "--parents" flag is not yet available in stable syntax
# See https://docs.docker.com/reference/dockerfile/#copy---parents

FROM python:3.11

WORKDIR /app

# cv2 (the python wrapper opencv-python) looks for cv2 (the shared object file) and
# does not find it despite https://pypi.org/project/opencv-python/#frequently-asked-questions
# "the packages are special wheel binary packages and they already contain statically built OpenCV binaries"?
RUN apt-get update
RUN apt-get install python3-opencv -y

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -U openmim
RUN mim install mmengine
RUN mim install "mmcv>=2.0.0" # Do expect this mmcv install to take like 10 minutes
RUN pip install "mmsegmentation>=1.0.0"

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --parents flask-app.py /models /tools .

CMD flask --app flask-app.py run --host=0.0.0.0
