import retro

def main():
    env = retro.make(game='Jackal-Nes')
    obs = env.reset()
    act = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    while True:
        '''
        0 小子弹
        1 无
        2 无
        3 无
        4 上
        5 下
        6 左
        7 右
        8 手榴弹/火箭炮
        '''
        #act = env.action_space.sample()
        i = 8
        act[i] = 1
        obs, rew, done, info = env.step(act)
        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
