# ==================================
# Base: CUDA 12.9 + Ubuntu 24.04
# ==================================
FROM nvcr.io/nvidia/tensorrt:24.02-py3

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# ==================================
# 1. System Dependencies
# ==================================
RUN apt update && apt install -y \
    curl gnupg lsb-release \
    python3 python3-pip python3-venv \
    software-properties-common \
    git build-essential \
    libopencv-dev python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# ==================================
# 2. Python Dependencies
# ==================================

RUN pip install --upgrade pip

# ML + colcon inside venv
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    opencv-python-headless==4.8.1.78 \
    colcon-common-extensions

# ==================================
# 3. Add ROS 2 Jammy Repository
# ==================================
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu jammy main" \
    > /etc/apt/sources.list.d/ros2.list


# ==================================
# 5. Install ROS + Gazebo + Bridge
# ==================================
RUN apt update && apt install -y \
    ros-humble-ros-base \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-vision-msgs \
    ros-humble-rmw-cyclonedds-cpp \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init || true
RUN rosdep update

# ==================================
# 6. ROS Workspace
# ==================================
WORKDIR /robotics_ws
COPY src /robotics_ws/src

RUN source /opt/ros/humble/setup.bash && \
    colcon build

# ==================================
# 7. Auto Source
# ==================================
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /robotics_ws/install/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
