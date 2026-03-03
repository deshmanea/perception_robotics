# ==================================
# Base: CUDA 12.9 + Ubuntu 24.04
# ==================================
FROM nvidia/cuda:12.9.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# ==================================
# 1. System Dependencies
# ==================================
RUN apt update && apt install -y \
    curl \
    gnupg \
    lsb-release \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# ==================================
# 2. Create Python Virtual Env
# ==================================
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip

# ML + colcon inside venv
RUN pip install --no-cache-dir \
    colcon-common-extensions \
    torch \
    torchvision \
    torchaudio \
    ultralytics

# ==================================
# 3. Add ROS 2 Jazzy Repository
# ==================================
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu noble main" \
    > /etc/apt/sources.list.d/ros2.list

# ==================================
# 4. Add Gazebo Harmonic Repository
# ==================================
RUN curl -sSL https://packages.osrfoundation.org/gazebo.gpg \
    | gpg --dearmor -o /usr/share/keyrings/gazebo-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/gazebo-archive-keyring.gpg] \
    http://packages.osrfoundation.org/gazebo/ubuntu-stable noble main" \
    > /etc/apt/sources.list.d/gazebo-stable.list

# ==================================
# 5. Install ROS + Gazebo + Bridge
# ==================================
RUN apt update && apt install -y \
    ros-jazzy-ros-base \
    ros-jazzy-ros-gz \
    ros-jazzy-ros-gz-bridge \
    gz-harmonic \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init || true
RUN rosdep update

# ==================================
# 6. ROS Workspace
# ==================================
WORKDIR /robotics_ws
COPY src /robotics_ws/src

RUN source /opt/ros/jazzy/setup.bash && \
    colcon build

# ==================================
# 7. Auto Source
# ==================================
RUN echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc && \
    echo "source /robotics_ws/install/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
