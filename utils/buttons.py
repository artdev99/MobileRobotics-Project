async def wait_for_start_button(node, client):
    await node.wait_for_variables({"button.forward"})
    await client.sleep(0.05)
    return (node.v.button.forward)
    
async def check_stop_button(node, client):
    await node.wait_for_variables({"button.center"})
    if (node.v.button.center == 1): 
        print("stopping")
        return True
    else :
        return False
