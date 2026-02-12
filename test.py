import retro

def main():
    env = retro.make(game='Jackal-Nes')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(env.action_space.sample())
        print(info)
        env.render()
        if done:
            obs = env.reset()
    env.close()


if __name__ == "__main__":
    main()
