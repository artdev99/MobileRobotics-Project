SPEED_LIMIT = 500 #PWM

async def set_motors(node, v_l, v_r): #v_l and v_r in PWM
    v_ml = limit_speed(v_ml)
    v_mr = limit_speed(v_mr)

    v_ml = int(v_ml)  #ensure integer type
    v_mr = int(v_mr)  #ensure integer type

    v_m = {
        "motor.left.target": [v_ml],
        "motor.right.target": [v_mr],
    }
    await node.set_variables(v_m)


def limit_speed(v):
    if(v>SPEED_LIMIT) :
        v=SPEED_LIMIT
    if(v<-SPEED_LIMIT) :
        v=-SPEED_LIMIT
    return v
        