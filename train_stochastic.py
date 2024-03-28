# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Run off-policy evaluation training loop."""

import torch
import gc
import json
import os
import pickle
from absl import app
from absl import flags
from absl import logging
# import d4rl  # pylint: disable=unused-import  
import gym
from gym.wrappers import time_limit
import numpy as np
import tensorflow as tf
from tf_agents.environments import gym_wrapper
from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
import tqdm

from policy_eval import utils
from policy_eval.actor import Actor
from policy_eval.behavior_cloning import BehaviorCloning
from policy_eval.dataset import D4rlDataset
from policy_eval.dataset import Dataset
from policy_eval.dual_dice import DualDICE
from policy_eval.model_based import ModelBased
from policy_eval.q_fitter import QFitter

from runner import Runner
import datetime as dt
import options

EPS = np.finfo(np.float32).eps
FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'Ant-v2',
                    'Environment for training/evaluation.')
flags.DEFINE_bool('d4rl', False, 'Whether to use D4RL envs and datasets.')
flags.DEFINE_string('d4rl_policy_filename', None,
                    'Path to saved pickle of D4RL policy.')
flags.DEFINE_integer('seed', 0, 'Fixed random seed for training.')
flags.DEFINE_float('lr', 3e-4, 'Critic learning rate.')
flags.DEFINE_float('weight_decay', 1e-5, 'Weight decay.')
flags.DEFINE_float('behavior_policy_std', 0,
                   'Noise scale of behavior policy.')
flags.DEFINE_float('target_policy_std', 1.25, 'Noise scale of target policy.')
flags.DEFINE_integer('num_trajectories', 50, 'Number of trajectories.')
flags.DEFINE_integer('sample_batch_size', 256, 'Batch size.')
flags.DEFINE_integer(
    'num_mc_episodes', 128,
    'Number of episodes to unroll to estimate Monte Carlo returns.')
flags.DEFINE_integer('num_updates', 1_000_00, 'Number of updates.')
flags.DEFINE_integer('eval_interval', 10_000, 'Logging interval.')
flags.DEFINE_integer('log_interval', 10_000, 'Logging interval.')
flags.DEFINE_float('discount', 0.99, 'Discount used for returns.')
flags.DEFINE_float('tau', 0.005,
                   'Soft update coefficient for the target network.')
flags.DEFINE_string('save_dir', '/tmp/policy_eval/',
                    'Directory to save results to.')
flags.DEFINE_string(
    'data_dir',
    '/root/CSBI/policy_eval/trajectory_datasets',
    'Directory with data for evaluation.')
flags.DEFINE_boolean('normalize_states', True, 'Whether to normalize states.')
flags.DEFINE_boolean('normalize_rewards', True, 'Whether to normalize rewards.')
flags.DEFINE_boolean('bootstrap', True,
                     'Whether to generated bootstrap weights.')
flags.DEFINE_enum('algo', 'mb', ['fqe', 'dual_dice', 'mb', 'iw', 'dr'],
                  'Algorithm for policy evaluation.')
flags.DEFINE_float('noise_scale', 0.5, 'Noise scale')
flags.DEFINE_string('models_dir', "policy_eval/data", 'Model to load for evaluation.')
flags.DEFINE_integer('horizon', 1000, 'Horizon of MDP')

def make_hparam_string(json_parameters=None, **hparam_str_dict):
  if json_parameters:
    for key, value in json.loads(json_parameters).items():
      if key not in hparam_str_dict:
        hparam_str_dict[key] = value
  return ','.join([
      '%s=%s' % (k, str(hparam_str_dict[k]))
      for k in sorted(hparam_str_dict.keys())
  ])


def main(_):
  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)

  hparam_str = make_hparam_string(
      seed=FLAGS.seed, env_name=FLAGS.env_name)
  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.save_dir, 'tb', hparam_str))
  summary_writer.set_as_default()

  if FLAGS.d4rl:
    d4rl_env = gym.make(FLAGS.env_name)
    gym_spec = gym.spec(FLAGS.env_name)
    if gym_spec.max_episode_steps in [0, None]:  # Add TimeLimit wrapper.
      gym_env = time_limit.TimeLimit(d4rl_env, max_episode_steps=1000)
    else:
      gym_env = d4rl_env
    gym_env.seed(FLAGS.seed)
    env = tf_py_environment.TFPyEnvironment(gym_wrapper.GymWrapper(gym_env))

  if FLAGS.d4rl:
    with tf.io.gfile.GFile(FLAGS.d4rl_policy_filename, 'rb') as f:
      policy_weights = pickle.load(f)
    actor = utils.D4rlActor(env, policy_weights,
                            is_dapg='dapg' in FLAGS.d4rl_policy_filename)
  else:
    env = suite_mujoco.load(FLAGS.env_name)
    env.seed(FLAGS.seed)
    env = tf_py_environment.TFPyEnvironment(env)
    actor = Actor(env.observation_spec().shape[0], env.action_spec())
    
    # print(behavior_dataset.model_filename)
    actor.load_weights(os.getcwd()+"/"+os.path.join(FLAGS.models_dir, 'DM-' + FLAGS.env_name,
                                str(0), '1000000'))

  print("collecting trajectories under behavior policy")
  print("="*80)
  dataset, reward_max, reward_min= utils.collect_stochastic_env_dataset(env, actor, std = FLAGS.behavior_policy_std,
                                                num_episodes = FLAGS.num_trajectories, horizon = FLAGS.horizon, max_length = 1000, noise_scale = FLAGS.noise_scale)


  tf_dataset = dataset.with_uniform_sampling(FLAGS.sample_batch_size)
  tf_dataset_iter = iter(tf_dataset)

  print("="*80)
  print(f"\t\tDSBTraining start at {dt.datetime.now().strftime('%m_%d_%Y_%H%M%S')}")
  print("="*80)
  print("setting configurations...")
  opt = options.set("mdp")
  run = Runner(opt, dataset)
  run.sb_alternate_imputation_train(opt)

  # utils.sample_dynamic(FLAGS.env_name, FLAGS.seed, actor,std=0, num_samples = 100, DSB=run, opt = opt)

  print("baseline algorithm training")
  print("="*80)

  # baseline_model_names = [ 'dual_dice','fqe', 'mb']
  # baseline_model_names = ["fqe", "dual_dice"]
  baseline_model_names = ["fqe", "mb"]
  # baseline_model_names = ["mb"]
  models= []

  min_reward = np.min(dataset.rewards).astype(np.float32)
  max_reward = np.max(dataset.rewards).astype(np.float32)
  min_state = np.min(dataset.states, 0).astype(np.float32)
  max_state = np.max(dataset.states, 0).astype(np.float32)

  @tf.function
  def update_step(model, model_name):
    (states, actions, next_states, rewards, masks, weights) = next(tf_dataset_iter)
    rewards = tf.squeeze(rewards)
    _, initial_actions, _ = actor(dataset.initial_states.astype(np.float32), FLAGS.target_policy_std)
    _, next_actions, _ = actor(next_states, FLAGS.target_policy_std)  

    if model_name == "fqe":
      model.update(states, actions, next_states, next_actions, rewards, masks,
                   weights, FLAGS.discount, min_reward, max_reward)
    elif model_name == "mb":
      model.update(states, actions, next_states, rewards, masks,
                   weights)
    elif model_name == "dual_dice":
      model.update(dataset.initial_states.astype(np.float32), initial_actions,
                   dataset.initial_weights.astype(np.float32), states, actions,
                   next_states, next_actions, masks, weights, FLAGS.discount)
    else:
      raise NotImplementedError


  for model_name in baseline_model_names:
    if model_name == "fqe":
      model = QFitter(env.observation_spec().shape[0],
                      env.action_spec().shape[0], FLAGS.lr, FLAGS.weight_decay,
                      FLAGS.tau)
    elif model_name == "mb":
      model = ModelBased(env.observation_spec().shape[0],
                        env.action_spec().shape[0], learning_rate=FLAGS.lr,
                        weight_decay=FLAGS.weight_decay)
    elif model_name == "dual_dice":
      model = DualDICE(env.observation_spec().shape[0],
                       env.action_spec().shape[0], FLAGS.weight_decay)
    else:
      raise NotImplementedError
    
    print(f"training {model_name}...")
    gc.collect()
    for i in tqdm.tqdm(range(FLAGS.num_updates), desc='Running Training'):
      update_step(model, model_name)
    print("done training algorithm ", model_name)
    models.append(model)


  stds = [0.1, 0.25, 0.5, 1.25, 2.5]
  print("begin evaluation")
  print("="*80)
  for std in stds:
    print("evaluating target policy std ", std)
    target_mc_return = utils.collect_stochastic_env_dataset(env, actor, std, FLAGS.num_mc_episodes, horizon = FLAGS.horizon, max_length = 1000, discount = FLAGS.discount, noise_scale = FLAGS.noise_scale,
                                                               reward_max=reward_max, reward_min=reward_min)
    print("target policy return is ", target_mc_return.numpy())
    # if FLAGS.env_name != "Ant-v2":
    DSB_return = utils.estimate_returns(env, actor, std, FLAGS.num_trajectories, horizon = FLAGS.horizon, initial_states = dataset.initial_states, DSB = run, opt = opt, discount = FLAGS.discount)
    # else:
    #   DSB_return = utils.estimate_returns(env, actor, std, 100, horizon = FLAGS.horizon, initial_states = dataset.initial_states[:100], DSB = run, opt = opt, discount = FLAGS.discount)
    print("DSB return is ", DSB_return)
    print("DSB bias is", abs(DSB_return - target_mc_return).numpy())

    for i in range(len(baseline_model_names)):
      model_name = baseline_model_names[i]
      model = models[i]
      if model_name == "fqe":
        pred_returns = model.estimate_returns(dataset.initial_states.astype(np.float32),
                                              dataset.initial_weights.astype(np.float32),
                                              actor, std)
      elif model_name == "mb":
        pred_returns = model.estimate_returns(dataset.initial_states,
                                              dataset.initial_weights,
                                              actor,
                                              std,
                                              FLAGS.discount,
                                              min_reward, max_reward,
                                              min_state, max_state,
                                              horizon = FLAGS.horizon)
      elif model_name == "dual_dice":
        pred_returns, pred_ratio = model.estimate_returns(iter(tf_dataset))
        tf.summary.scalar('train/pred ratio', pred_ratio, step=i)

      print(model_name, "pred returns is ", pred_returns.numpy())
      print(model_name, "bias is ", abs(pred_returns - target_mc_return).numpy())

if __name__ == '__main__':
  app.run(main)
