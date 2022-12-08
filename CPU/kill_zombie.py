import os
print("kill_zonbie_process")
os.system("ps -ef | grep defunct | awk '{print $3}' | xargs kill -9")
