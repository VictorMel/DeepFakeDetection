import os
import subprocess
import grp

# Function to check if the user is in the groups 
def check_and_add_user_to_groups(user):
    groups_to_check = ["render", "video"]
    
    try:
        # Get the current groups of the user
        result = subprocess.run(['groups', user], capture_output=True, text=True, check=True)
        user_groups = result.stdout.strip().split(':')[-1].split()
        
        for group in groups_to_check:
            if group not in user_groups:
                print(f"User {user} is not in the {group} group. Adding to {group} group.")
                subprocess.run(['sudo', 'usermod', '-aG', group, user], check=True)
            else:
                print(f"User {user} is already in the {group} group.")
                
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

# Function to check if a package is installed
def is_package_installed(package_name):
    try:
        result = subprocess.run(['dpkg', '-l', package_name], check=True, capture_output=True, text=True)
        return package_name in result.stdout
    except subprocess.CalledProcessError:
        return False

# Function to install a package
def install_package(package_name):
    try:
        subprocess.run(['sudo', 'apt', 'install', '-y', package_name], check=True)
        print(f'{package_name} installed successfully.')
    except subprocess.CalledProcessError as e:
        print(f'Failed to install {package_name}: {e}')
        
        
# Function to check if a package is installed and print its version
def check_package_version(package_name):
    try:
        result = subprocess.run(['dpkg', '-s', package_name], check=True, capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if line.startswith('Version:'):
                print(f'{package_name} version: {line.split(":")[1].strip()}')
                return True
        return False
    except subprocess.CalledProcessError:
        return False