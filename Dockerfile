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

# Model files too large for GitHub; pull from Google Drive.
# (The image is built and hosted privately on Red Hat OpenShift.)
ADD https://drive.usercontent.google.com/download?id=14KPOwVHZB9_V4u4Ih2MxgrgdQl-Pk_81&export=download&confirm=t models/convnext-base_upernet_8xb2-amp-160k_tem-512x512_cpu.py
ADD https://drive.usercontent.google.com/download?id=1b5NeYgmJFd7Z1redf3TJUGL8_aol1oR4&export=download&confirm=t models/convnext-base_upernet_8xb2-amp-160k_tem-512x512.py
# Do expect this GB-scale download to take 1-2 minutes
ADD https://drive.usercontent.google.com/download?id=1Z0OVd7i5zo8IeNeRtLQcPtHoF6QH0ZXZ&export=download&confirm=t models/iter_64000.pth

COPY tools/inference.py tools/measurement4mask.py tools/
COPY flask-app.py .

# Set workdir owned by root group and readable by group
# https://www.redhat.com/en/blog/a-guide-to-openshift-and-uids
RUN chown --recursive :0 .
RUN chmod --recursive g+r .

CMD flask --app flask-app.py run --host=0.0.0.0
