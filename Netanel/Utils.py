import os
import subprocess
import grp

# Function to check if the user is in the "render" group

def user_in_render_group(user):
    try:
        groups = [g.gr_name for g in grp.getgrall() if user in g.gr_mem]
        gid = os.getgid()
        group = grp.getgrgid(gid).gr_name
        return group == 'render' or 'render' in groups
    except KeyError:
        return False

# Function to add user to "render" group
def add_user_to_render_group(user):
    try:
        subprocess.run(['sudo', 'usermod', '-aG', 'render', user], check=True)
        print(f'User {user} added to render group. Please log out and log back in for changes to take effect.')
    except subprocess.CalledProcessError as e:
        print(f'Failed to add user {user} to render group: {e}')

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