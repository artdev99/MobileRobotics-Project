from time import sleep
from tdmclient import ClientAsync, aw

CONNECTED = 1
UNLOCKED = 2
BUSY = 3
LOCKED = 4
DISCONNECTED = 5


def get_status(node):
    status = node.status
    if status == CONNECTED:
        return "Connected"
    if status == UNLOCKED:
        return "Unlocked"
    if status == BUSY:
        return "Busy"
    if status == LOCKED:
        return "Locked"
    if status == DISCONNECTED:
        return "Disconnected"
    else:
        return "Unknown"

def lock_node(node):
    aw(node.lock())
    sleep(0.1)
    get_status(node)
               
def unlock_node(node):
    aw(node.unlock())
    sleep(0.1)
    get_status(node)

def init_node():
    client = ClientAsync()
    node = aw(client.wait_for_node())
    sleep(0.1)
    
    if node.status != 2:
        print("Please unlock the thymio in Aseba Studio or check connection")
    
    aw(node.wait_for_variables())
    
    try:
        aw(node.lock())
        sleep(0.1)
        aw(node.unlock())
    except:
        print("Please unlock Thymio in Aseba Studio")

    sleep(0.1)
    print(node.props["name"], get_status(node))