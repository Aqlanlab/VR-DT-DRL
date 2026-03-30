#!/bin/bash
# VM Simulation System Setup Script
# Run this script on Ubuntu 18.04 VM to install all dependencies

set -e  # Exit on any error

echo "🤖 Starting VM Simulation System Setup..."
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Ubuntu 18.04
print_status "Checking Ubuntu version..."
if [[ $(lsb_release -rs) != "18.04" ]]; then
    print_warning "This script is designed for Ubuntu 18.04. Current version: $(lsb_release -rs)"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Update system packages
print_status "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install essential packages
print_status "Installing essential packages..."
sudo apt install -y \
    build-essential \
    cmake \
    git \
    vim \
    curl \
    wget \
    python3-pip \
    python3-dev \
    python3-setuptools \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release

# Check if ROS Melodic is installed
if ! command -v roscore &> /dev/null; then
    print_status "Installing ROS Melodic..."
    
    # Add ROS repository
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    
    # Add ROS key
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
    
    # Update package index
    sudo apt update
    
    # Install ROS Melodic Desktop Full
    sudo apt install -y ros-melodic-desktop-full
    
    # Initialize rosdep
    if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
        sudo rosdep init
    fi
    rosdep update
    
    # Setup environment
    echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
    source /opt/ros/melodic/setup.bash
    
    print_status "ROS Melodic installed successfully!"
else
    print_status "ROS Melodic already installed"
fi

# Install additional ROS packages
print_status "Installing additional ROS packages..."
sudo apt install -y \
    ros-melodic-cv-bridge \
    ros-melodic-image-transport \
    ros-melodic-compressed-image-transport \
    ros-melodic-camera-info-manager \
    ros-melodic-robot-state-publisher \
    ros-melodic-joint-state-publisher \
    ros-melodic-joint-state-publisher-gui \
    ros-melodic-xacro \
    ros-melodic-tf2-tools \
    ros-melodic-rqt \
    ros-melodic-rqt-common-plugins \
    ros-melodic-rviz

# Install Python packages
print_status "Installing Python packages..."
pip3 install --user -r requirements.txt

# Create catkin workspace if it doesn't exist
if [ ! -d "$HOME/catkin_ws" ]; then
    print_status "Creating catkin workspace..."
    mkdir -p ~/catkin_ws/src
    cd ~/catkin_ws/
    catkin_make
    echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
    source ~/catkin_ws/devel/setup.bash
else
    print_status "Catkin workspace already exists"
fi

# Copy ROS package to catkin workspace
print_status "Setting up ROS package..."
PACKAGE_NAME="ur3_simulation"
if [ ! -d "$HOME/catkin_ws/src/$PACKAGE_NAME" ]; then
    mkdir -p ~/catkin_ws/src/$PACKAGE_NAME
    cp -r . ~/catkin_ws/src/$PACKAGE_NAME/
    cd ~/catkin_ws/
    catkin_make
    source ~/catkin_ws/devel/setup.bash
fi

# Check if Webots is installed
if ! command -v webots &> /dev/null; then
    print_status "Webots not found. Please install Webots R2023a manually:"
    print_warning "1. Download from: https://github.com/cyberbotics/webots/releases/tag/R2023a"
    print_warning "2. Extract to /opt/webots"
    print_warning "3. Add to PATH: export PATH=\$PATH:/opt/webots"
    print_warning "4. Set WEBOTS_HOME: export WEBOTS_HOME=/opt/webots"
else
    print_status "Webots is already installed"
fi

# Set up network configuration
print_status "Configuring network settings..."
VM_IP=$(ip route get 1 | awk '{print $7; exit}')
print_status "Detected VM IP: $VM_IP"

# Create config directory if it doesn't exist
mkdir -p config

# Update network config with detected IP
if [ -f "config/network_config.yaml" ]; then
    sed -i "s/vm_ip: .*/vm_ip: \"$VM_IP\"/" config/network_config.yaml
    print_status "Updated network configuration with VM IP: $VM_IP"
fi

# Set permissions for scripts
print_status "Setting script permissions..."
find . -name "*.py" -exec chmod +x {} \;
find . -name "*.sh" -exec chmod +x {} \;

# Create log directories
print_status "Creating log directories..."
mkdir -p data/logs
mkdir -p data/training_data
mkdir -p data/episodes

# Test ROS installation
print_status "Testing ROS installation..."
source /opt/ros/melodic/setup.bash
if roscore --help &> /dev/null; then
    print_status "ROS installation test passed ✅"
else
    print_error "ROS installation test failed ❌"
    exit 1
fi

# Test Python imports
print_status "Testing Python imports..."
python3 -c "
try:
    import rospy
    import cv2
    import numpy as np
    import yaml
    import socket
    import json
    print('All Python imports successful ✅')
except ImportError as e:
    print(f'Python import error: {e} ❌')
    exit(1)
"

# Final instructions
echo ""
echo "🎉 VM Simulation System setup complete!"
echo "======================================"
echo ""
print_status "Next steps:"
echo "1. Install Webots R2023a if not already installed"
echo "2. Update config/network_config.yaml with your host IP"
echo "3. Test the system with: python3 src/simulation_client.py"
echo ""
print_status "Your VM IP address: $VM_IP"
print_warning "Make sure to use this IP in your host system configuration!"
echo ""
print_status "To start the system:"
echo "   roslaunch vm_simulation_system ur3_simulation.launch"
echo ""

# Source the setup files
source ~/.bashrc 2>/dev/null || true

print_status "Setup completed successfully! 🚀"
