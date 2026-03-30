from controller import Robot

def main():
    # Initialize the Robot instance
    robot = Robot()

    # Get the basic time step of the simulation
    timestep = int(robot.getBasicTimeStep())

    # Initialize the motors using the exact names from our PROTO
    finger_1 = robot.getDevice('finger_1_motor')
    finger_2 = robot.getDevice('finger_2_motor')

    # Initialize the position sensors
    sensor_1 = robot.getDevice('finger_1_sensor')
    sensor_2 = robot.getDevice('finger_2_sensor')
    sensor_1.enable(timestep)
    sensor_2.enable(timestep)

    # Set the gripping force (Hand-E max is 130N)
    finger_1.setAvailableForce(130.0)
    finger_2.setAvailableForce(130.0)
    
    # Set a realistic closing speed (e.g., 50 mm/s)
    finger_1.setVelocity(0.05)
    finger_2.setVelocity(0.05)

    # Variables to track our open/close timer
    timer = 0
    is_closed = False

    print("Starting Robotiq Hand-E test sequence...")
    print("Initial state: Fully Open (0.0 m)")

    # Main control loop
    while robot.step(timestep) != -1:
        timer += timestep
        
        # Toggle the gripper every 3000 milliseconds (3 seconds)
        if timer >= 3000:
            timer = 0
            is_closed = not is_closed
            
            if is_closed:
                print("Closing Gripper... (Target: 0.025 m)")
                # Drives the fingers inward
                finger_1.setPosition(0.025)
                finger_2.setPosition(0.025)
            else:
                print("Opening Gripper... (Target: 0.0 m)")
                # Returns the fingers to their default max open position
                finger_1.setPosition(0.0)
                finger_2.setPosition(0.0)

if __name__ == '__main__':
    main()