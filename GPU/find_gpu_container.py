import sys
import docker
import psutil
import subprocess

def get_gpu_info():
    chunk = subprocess.check_output('nvidia-smi', shell=True).decode().split('=============================================================================')[-1].split("\n")
    gpu_info = []
    for part in chunk:
        part = ' '.join(part.split()).split(' ')
        if len(part) > 5:
            gpu_info.append([int(part[1]), int(part[4])])
    return gpu_info

def find_pid_info(gpu_pid):
    init_pid = int(gpu_pid)
    max_trial = 15
    try:
        current_p = psutil.Process(init_pid) 
    except:
        print('please enter the correct pid!')
        sys.exit()

    i = 0
    while(True):
        if i == max_trial:
            print('not exist!')
            break
        i += 1

        parent_pid = current_p.ppid()
        parent_p = psutil.Process(parent_pid)
        parent_cmd = ' '.join(parent_p.cmdline())

        if 'containerd-shim' in parent_cmd: ## find keyword
            return find_docker_name(parent_cmd)
            break
        else:
            current_p = parent_p

def find_docker_name(parent_cmd):
    client = docker.from_env()
    containers = client.containers.list()
    container_info = [[container.id, container.name] for container in containers]
    for container_info_ in container_info:
        if container_info_[0] in parent_cmd:
            return container_info_[1]
        
if __name__ == '__main__':
    gpu_info = get_gpu_info()
    if len(gpu_info) > 0:
        for gpu_ in gpu_info:
            print(str(gpu_[0]) + ' ' + str(gpu_[1]) + ' ' + str(find_pid_info(gpu_[1])))
