import pickle
import tensorflow as tf
import numpy as np
import tf_utils as U

"""Implementation of Behavioral Cloning on supervised data (state, action) collected
by performing rollouts with an expert policy.
Those rollouts are contained in pickle files available in the `trajectories/` folder.
Trajectories are pickled by running the `run_expert` script with the appropriate environment.
"""


def train(ob_dim, act_dim, hid_size, lr, niter, obs, acts, ops, ckpt):
    # Initialize the seesion and variables
    U.single_threaded_session().__enter__()
    U.initialize()

    # Unpack the operators
    train_step, compute_loss, predict = ops

    for i in range(int(niter)):
        train_step(obs, acts)
        l = compute_loss(obs, acts)
        # For reference, usual way of training and obtaining the loss
        # _, l = sess.run([train_op, loss], feed_dict={
        #     ob: obs,
        #     act: acts
        # })
        if i % 100 == 0:
            print("--- Loss: {0} ---".format(l))

    # Save the variables to disk
    U.save_state(ckpt)
    print("Model saved in file: {0}.".format(ckpt))


def roll(nepisodes, maxnsteps, ops, envname, ckpt):
    # Initialize the session and variables
    U.single_threaded_session().__enter__()
    U.initialize()
    U.load_state(ckpt)
    print("Model loaded.")

    # Unpack the operators
    _, _, predict = ops

    import gym
    env = gym.make(envname)
    print(env.action_space)
    print(env.observation_space)
    for e in range(nepisodes):
        print("--- Episode #{0} ---".format(e))
        observ = env.reset()
        done = False
        steps = 0
        while not done and steps < maxnsteps:
            if steps % 100 == 0:
                print("--- Step #{0} ---".format(steps))
            action = predict(np.reshape(observ, (1, 11)))
            observ, r, done, _ = env.step(action)
            steps += 1
            env.render()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--envname", type=str)
    parser.add_argument("--expert_trajectories_file", type=str)
    parser.add_argument("--model_storage_file", type=str)
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()

    # Unpickle the expert trajectories
    with open(args.expert_trajectories_file, "rb") as f:
        expert_trajectories = pickle.loads(f.read())
    obs = expert_trajectories["observations"]
    print("--- OBS shape: {0} ---".format(obs.shape))
    print("First row of observations: {0}".format(obs[0, :]))
    acts = expert_trajectories["actions"].reshape(
        (expert_trajectories["actions"].shape[0], expert_trajectories["actions"].shape[2]))
    print("--- ACTS shape: {0} ---".format(acts.shape))
    print("First row of actions: {0}".format(acts[0, :]))

    # Hyperparameters
    # Behavioral Cloning (regression)
    ob_dim = 11
    act_dim = 3
    hid_size = 64
    lr = 1e-3
    niter = 1e4
    # Rollouts
    nepisodes = 100
    maxnsteps = 5000

    # Define the regression model
    ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[None, ob_dim])
    act = U.get_placeholder(name="act", dtype=tf.float32, shape=[None, act_dim])
    last_out = ob
    last_out = U.leaky_relu(
        U.dense(last_out, hid_size, "dense0", weight_init=U.normc_initializer(1.0)),
        leak=0.2)
    pred_act = U.dense(last_out, act_dim, "dense1", weight_init=U.normc_initializer(1.0))

    loss = 0.5 * U.sum(tf.square(pred_act - act))  # could also use Huber loss
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)

    # Create the operators
    train_step = U.function([ob, act], train_op)
    compute_loss = U.function([ob, act], loss)
    predict = U.function([ob], pred_act)
    # Pack the operators
    ops = [train_step, compute_loss, predict]

    # Train the regression model (only if the flag is up)
    if args.train:
        train(ob_dim, act_dim, hid_size, lr, niter, obs, acts, ops, args.model_storage_file)

    # Do some rollouts with the policy learned via regression (BC)
    roll(nepisodes, maxnsteps, ops, args.envname, args.model_storage_file)


if __name__ == "__main__":
    # executes only if run as a script
    main()
