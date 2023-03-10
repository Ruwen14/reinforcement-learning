if __name__ == '__main__':
    cur_reward = 10
    discount_factor = 0.99
    next_state_value = -30
    cur_state_value = -345

    delta_direct = cur_reward + discount_factor * next_state_value - cur_state_value

    target_value = cur_reward + discount_factor * next_state_value

    delta_indicrect = target_value - cur_state_value


    print(delta_direct ,delta_indicrect)