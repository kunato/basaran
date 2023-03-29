FROM hyperonym/basaran:0.13.4

# Set working directory
WORKDIR /app

# Install extra dependencies
RUN pip install icetk cpm_kernels

# Download the model to be bundled
RUN python utils/download.py THUDM/chatglm-6b /model refs/pr/7

# Provide default environment variables
ENV MODEL="/model"
ENV MODEL_REVISION="refs/pr/7"
ENV MODEL_LOCAL_FILES_ONLY="true"
ENV MODEL_TRUST_REMOTE_CODE="true"
ENV MODEL_HALF_PRECISION="true"
ENV SERVER_MODEL_NAME="THUDM/chatglm-6b"
