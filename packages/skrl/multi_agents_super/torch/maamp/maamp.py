from typing import Any, Callable, Mapping, Optional, Tuple, Union,Sequence

import copy
import itertools
import math
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.multi_agents_super.torch import MultiAgentSuper
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from collections import defaultdict

MAAMP_DEFAULT_CONFIG = {

  "PPO" :{
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 1e-3,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,      # entropy loss scaling factor
    "value_loss_scale": 1.0,        # value loss scaling factor

    "kl_threshold": 0,              # KL divergence threshold for early stopping

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
    },


   "AMP":{
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 6,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 5e-5,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})
    "amp_state_preprocessor": None,         # AMP state preprocessor class (see skrl.resources.preprocessors)
    "amp_state_preprocessor_kwargs": {},    # AMP state preprocessor's kwargs (e.g. {"size": env.amp_observation_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.0,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,          # entropy loss scaling factor
    "value_loss_scale": 2.5,            # value loss scaling factor
    "discriminator_loss_scale": 5.0,    # discriminator loss scaling factor

    "amp_batch_size": 512,                  # batch size for updating the reference motion dataset
    "task_reward_weight": 0.0,              # task-reward weight (wG)
    "style_reward_weight": 1.0,             # style-reward weight (wS)
    "discriminator_batch_size": 0,          # batch size for computing the discriminator loss (all samples if 0)
    "discriminator_reward_scale": 2,                    # discriminator reward scaling factor
    "discriminator_logit_regularization_scale": 0.05,   # logit regularization scale factor for the discriminator loss
    "discriminator_gradient_penalty_scale": 5,          # gradient penalty scaling factor for the discriminator loss
    "discriminator_weight_decay_scale": 0.0001,         # weight decay scaling factor for the discriminator loss

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
   }
}
# 

class MAAMP(MultiAgentSuper):
    def __init__(
        self,
        possible_agents: Sequence[str],
        models: Mapping[str, Model],
        memories: Optional[Mapping[str, Memory]] = None,
        observation_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
        action_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
        motion_dataset: Optional[Memory] = None,
        reply_buffer: Optional[Memory] = None,
        collect_reference_motions: Optional[Callable[[int], torch.Tensor]] = None,
        collect_observation: Optional[Callable[[], torch.Tensor]] = None,
        # memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        # observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        # action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        # amp_observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        # motion_dataset: Optional[Memory] = None,
        # reply_buffer: Optional[Memory] = None,
        # collect_reference_motions: Optional[Callable[[int], torch.Tensor]] = None,
        # collect_observation: Optional[Callable[[], torch.Tensor]] = None,
    ) -> None:
        # 解析模式：PPO / AMP
        # self.mode = cfg.get("mode", "PPO").upper() if cfg else "PPO"
        # assert self.mode in ["PPO", "AMP"], f"Unsupported mode: {self.mode}"

        _cfg = copy.deepcopy(MAAMP_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policies = {uid: self.models[uid].get("policy", None) for uid in self.possible_agents}
        self.values = {uid: self.models[uid].get("value", None) for uid in self.possible_agents}
        self.discriminator = {uid: self.models[uid].get("discriminator", None) for uid in self.possible_agents}
        

        for uid in self.possible_agents:
            # checkpoint models
            self.checkpoint_modules[uid]["policy"] = self.policies[uid]
            self.checkpoint_modules[uid]["value"] = self.values[uid]
            self.checkpoint_modules[uid]["discriminator"] = self.discriminator[uid]

            # broadcast models' parameters in distributed runs
            if config.torch.is_distributed:
                logger.info(f"Broadcasting models' parameters")
                if self.policies[uid] is not None:
                    self.policies[uid].broadcast_parameters()
                    if self.values[uid] is not None and self.policies[uid] is not self.values[uid]:
                        self.values[uid].broadcast_parameters() 
                if self.discriminator[uid] is not None:
                    self.discriminator[uid].broadcast_parameters()      

        # configuration
        print("Configuration:")
        # print(self.cfg)
        self._ppo_learning_epochs = self.cfg["PPO"]["learning_epochs"]
        self._ppo_mini_batches = self.cfg["PPO"]["mini_batches"]
        self._ppo_rollouts = self.cfg["PPO"]["rollouts"]
        self._ppo_rollout = 0

        self._ppo_grad_norm_clip = self.cfg["PPO"]["grad_norm_clip"]
        self._ppo_ratio_clip = self.cfg["PPO"]["ratio_clip"]
        self._ppo_value_clip = self.cfg["PPO"]["value_clip"]
        self._ppo_clip_predicted_values = self.cfg["PPO"]["clip_predicted_values"]

        self._ppo_value_loss_scale = self.cfg["PPO"]["value_loss_scale"]
        self._ppo_entropy_loss_scale = self.cfg["PPO"]["entropy_loss_scale"]

        self._ppo_kl_threshold = self.cfg["PPO"]["kl_threshold"]

        self._ppo_learning_rate = self.cfg["PPO"]["learning_rate"]
        self._ppo_learning_rate_scheduler = self.cfg["PPO"]["learning_rate_scheduler"]
        self._ppo_learning_rate_scheduler_kwargs = self.cfg["PPO"]["learning_rate_scheduler_kwargs"]

        self._ppo_state_preprocessor = self.cfg["PPO"]["state_preprocessor"]
        self._ppo_state_preprocessor_kwargs = self.cfg["PPO"]["state_preprocessor_kwargs"]
        self._ppo_value_preprocessor = self.cfg["PPO"]["value_preprocessor"]
        self._ppo_value_preprocessor_kwargs = self.cfg["PPO"]["value_preprocessor_kwargs"]

        self._ppo_discount_factor = self.cfg["PPO"]["discount_factor"]
        self._ppo_lambda = self.cfg["PPO"]["lambda"]

        self._ppo_random_timesteps = self.cfg["PPO"]["random_timesteps"]
        self._ppo_learning_starts = self.cfg["PPO"]["learning_starts"]

        self._ppo_rewards_shaper = self.cfg["PPO"]["rewards_shaper"]
        self._ppo_time_limit_bootstrap = self.cfg["PPO"]["time_limit_bootstrap"]

        self._ppo_mixed_precision = self.cfg["PPO"]["mixed_precision"]

        ### amp configuration
        self._amp_learning_epochs = self.cfg["AMP"]["learning_epochs"]
        self._amp_mini_batches = self.cfg["AMP"]["mini_batches"]
        self._amp_rollouts = self.cfg["AMP"]["rollouts"]
        self._amp_rollout = 0

        self._amp_grad_norm_clip = self.cfg["AMP"]["grad_norm_clip"]
        self._amp_ratio_clip = self.cfg["AMP"]["ratio_clip"]
        self._amp_value_clip = self.cfg["AMP"]["value_clip"]
        self._amp_clip_predicted_values = self.cfg["AMP"]["clip_predicted_values"]

        self._amp_value_loss_scale = self.cfg["AMP"]["value_loss_scale"]
        self._amp_entropy_loss_scale = self.cfg["AMP"]["entropy_loss_scale"]
        self._amp_discriminator_loss_scale = self.cfg["AMP"]["discriminator_loss_scale"]

        self._amp_learning_rate = self.cfg["AMP"]["learning_rate"]
        self._amp_learning_rate_scheduler = self.cfg["AMP"]["learning_rate_scheduler"]

        
        self._amp_state_preprocessor = self.cfg["AMP"]["state_preprocessor"]
        self._amp_state_preprocessor_kwargs = self.cfg["AMP"]["state_preprocessor_kwargs"]
        self._amp_value_preprocessor = self.cfg["AMP"]["value_preprocessor"]
        self._amp_value_preprocessor_kwargs = self.cfg["AMP"]["value_preprocessor_kwargs"]
        self._amp_amp_state_preprocessor = self.cfg["AMP"]["amp_state_preprocessor"]
        self._amp_amp_state_preprocessor_kwargs = self.cfg["AMP"]["amp_state_preprocessor_kwargs"]


        self._amp_discount_factor = self.cfg["AMP"]["discount_factor"]
        self._amp_lambda = self.cfg["AMP"]["lambda"]

        self._amp_random_timesteps = self.cfg["AMP"]["random_timesteps"]
        self._amp_learning_starts = self.cfg["AMP"]["learning_starts"]

        self._amp_batch_size = self.cfg["AMP"]["amp_batch_size"]
        self._amp_task_reward_weight = self.cfg["AMP"]["task_reward_weight"]
        self._amp_style_reward_weight = self.cfg["AMP"]["style_reward_weight"]

        self._amp_discriminator_batch_size = self.cfg["AMP"]["discriminator_batch_size"]
        self._amp_discriminator_reward_scale = self.cfg["AMP"]["discriminator_reward_scale"]
        self._amp_discriminator_logit_regularization_scale = self.cfg["AMP"]["discriminator_logit_regularization_scale"]
        self._amp_discriminator_gradient_penalty_scale = self.cfg["AMP"]["discriminator_gradient_penalty_scale"]
        self._amp_discriminator_weight_decay_scale = self.cfg["AMP"]["discriminator_weight_decay_scale"]

        self._amp_rewards_shaper = self.cfg["AMP"]["rewards_shaper"]
        self._amp_time_limit_bootstrap = self.cfg["AMP"]["time_limit_bootstrap"]

        self._amp_mixed_precision = self.cfg["AMP"]["mixed_precision"]

        if observation_spaces is not None:
            self.amp_observation_space = observation_spaces['humanoid']
        else:
            print("observation is None!")
        self.motion_dataset = motion_dataset
        self.reply_buffer = reply_buffer
        self.collect_reference_motions = collect_reference_motions
        self.collect_observation = collect_observation


        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._ppo_mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._ppo_mixed_precision)

        # set up optimizer and learning rate scheduler
        self.optimizers = {}
        self.schedulers = {}

        print("Setting up optimizers and learning rate schedulers...")
        # for uid in self.possible_agents:
        #     print(f"  {uid}")

        ppo_policy = self.policies["exo"]
        ppo_value = self.values["exo"]
        # print(f"  {uid}: policy={policy} value={value}")
        if ppo_policy is not None and ppo_value is not None:
            if ppo_policy is ppo_value:
                optimizer = torch.optim.Adam(ppo_policy.parameters(), lr=self._ppo_learning_rate[uid])
            else:
                print(" self._ppo_learning_rate: ", self._ppo_learning_rate)
                optimizer = torch.optim.Adam(
                    itertools.chain(ppo_policy.parameters(), ppo_value.parameters()), lr=self._ppo_learning_rate
                )
            self.optimizers["exo"] = optimizer
            if self._ppo_learning_rate_scheduler is not None:
                self.schedulers["exo"] = self._ppo_learning_rate_scheduler(
                    optimizer, **self._ppo_learning_rate_scheduler_kwargs
                )
        self.checkpoint_modules["exo"]["optimizer"] = self.optimizers["exo"]


        # # set up automatic mixed precision
        # self._device_type = torch.device(device).type
        # if version.parse(torch.__version__) >= version.parse("2.4"):
        #     self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        # else:
        #     self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizer and learning rate scheduler
        if self.policies["humanoid"] is not None and self.values["humanoid"] is not None and self.discriminator["humanoid"] is not None:
            optimizer = torch.optim.Adam(
                itertools.chain(self.policies["humanoid"].parameters(),self.values["humanoid"].parameters(), self.discriminator["humanoid"].parameters()),
                lr=self._amp_learning_rate,
            )
            self.optimizers["humanoid"] = optimizer
            if self._amp_learning_rate_scheduler is not None:
                self.scheduler = self._amp_learning_rate_scheduler(
                    optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["humanoid"]["optimizer"] = optimizer

        # set up preprocessors

        if self._ppo_state_preprocessor is not None:
            self._ppo_state_preprocessor= self._ppo_state_preprocessor(**self._ppo_state_preprocessor_kwargs)
            self.checkpoint_modules["exo"]["state_preprocessor"] = self._ppo_state_preprocessor
        else:
            self._ppo_state_preprocessor = self._empty_preprocessor

        if self._ppo_value_preprocessor is not None:
            self._ppo_value_preprocessor = self._ppo_value_preprocessor(**self._ppo_value_preprocessor_kwargs)
            self.checkpoint_modules["exo"]["value_preprocessor"] = self._ppo_value_preprocessor
        else:
            self._ppo_value_preprocessor = self._empty_preprocessor

        if self._amp_state_preprocessor:
            self._amp_state_preprocessor = self._amp_state_preprocessor(**self._amp_state_preprocessor_kwargs)
            self.checkpoint_modules["humanoid"]["state_preprocessor"] = self._amp_state_preprocessor
        else:
            self._amp_state_preprocessor = self._empty_preprocessor

        if self._amp_value_preprocessor:
            self._amp_value_preprocessor = self._amp_value_preprocessor(**self._amp_value_preprocessor_kwargs)
            self.checkpoint_modules["humanoid"]["value_preprocessor"] = self._amp_value_preprocessor
        else:
            self._amp_value_preprocessor = self._empty_preprocessor

        if self._amp_amp_state_preprocessor:
            self._amp_amp_state_preprocessor = self._amp_amp_state_preprocessor(**self._amp_amp_state_preprocessor_kwargs)
            self.checkpoint_modules["humanoid"]["amp_state_preprocessor"] = self._amp_amp_state_preprocessor
        else:
            self._amp_state_preprocessor = self._empty_preprocessor
        
        print("__init__ finished")

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memories
        if self.memories:
            for uid in self.possible_agents:
                self.memories[uid].create_tensor(name="states", size=self.observation_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="actions", size=self.action_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="rewards", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="terminated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="truncated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="log_prob", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="values", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="returns", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="advantages", size=1, dtype=torch.float32)

                self.memories[uid].create_tensor(name="amp_states", size=self.amp_observation_space, dtype=torch.float32)
                self.memories[uid].create_tensor(name="next_values", size=1, dtype=torch.float32)

                # tensors sampled during training
                self._tensors_names = ["states", "actions", "log_prob", "values", "returns", "advantages","amp_states","next_values"]

        # create tensors for motion dataset and reply buffer
        if self.motion_dataset is not None:
            self.motion_dataset.create_tensor(name="states", size=self.amp_observation_space, dtype=torch.float32)
            self.reply_buffer.create_tensor(name="states", size=self.amp_observation_space, dtype=torch.float32)

            # initialize motion dataset
            for _ in range(math.ceil(self.motion_dataset.memory_size / self._amp_batch_size)):
                self.motion_dataset.add_samples(states=self.collect_reference_motions(self._amp_batch_size))
        # create temporary variables needed for storage and computation
        self._current_log_prob = []
        self._current_next_states = []
        self._current_states = None


        print("finish init")

    def act(self, states: Mapping[str, torch.Tensor], timestep: int, timesteps: int) -> torch.Tensor:
    # torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policies

        :param states: Environment's states
        :type states: dictionary of torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # # sample random actions
        # # TODO: fix for stochasticity, rnn and log_prob
        # if timestep < self._random_timesteps:
        #     return self.policy.random_act({"states": states}, role="policy")

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._amp_mixed_precision):

        # for uid in self.possible_agents:
        #     print(uid)
            data =[]

            preprocessed_state = self._ppo_state_preprocessor(states["exo"])
            output = self.policies["exo"].act({"states": preprocessed_state}, role="policy")


            data.append(output)

            if self._current_states is not None:
                preprocessed_state = self._amp_state_preprocessor(self._current_states)
            else: 
                preprocessed_state = self._amp_state_preprocessor(states["humanoid"])
            output = self.policies["humanoid"].act({"states": preprocessed_state}, role="policy")

            data.append(output)
            print("self.possible_agents: ",self.possible_agents)
            actions = {uid: d[0] for uid, d in zip(self.possible_agents, data)}
            log_prob = {uid: d[1] for uid, d in zip(self.possible_agents, data)}
            outputs = {uid: d[2] for uid, d in zip(self.possible_agents, data)}

            self._current_log_prob = log_prob
            print("actions:", actions)
        return actions, log_prob, outputs

    def record_transition(
        self,
        states: Mapping[str, torch.Tensor],
        actions: Mapping[str, torch.Tensor],
        rewards: Mapping[str, torch.Tensor],
        next_states: Mapping[str, torch.Tensor],
        terminated: Mapping[str, torch.Tensor],
        truncated: Mapping[str, torch.Tensor],
        infos: Mapping[str, Any],
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: dictionary of torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: dictionary of torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: dictionary of torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: dictionary of torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: dictionary of torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: dictionary of torch.Tensor
        :param infos: Additional information about the environment
        :type infos: dictionary of any supported type
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memories:
            self._current_next_states = next_states
            print(infos)
            amp_states = infos["amp_obs"]
            values = {}
            # reward shaping
            if self._ppo_rewards_shaper is not None:
                rewards["exo"] = self._ppo_rewards_shaper(rewards["exo"], timestep, timesteps)

            if self._amp_rewards_shaper is not None:
                rewards["humanoid"] = self._amp_rewards_shaper(rewards["humanoid"], timestep, timesteps)

            # compute values
            with torch.autocast(device_type=self._device_type, enabled=self._ppo_mixed_precision):
                
                values["exo"], _, _ = self.values["exo"].act(
                    {"states": self._ppo_state_preprocessor(states["exo"])}, role="value"
                )
                values["exo"] = self._ppo_value_preprocessor(values["exo"], inverse=True)

            with torch.autocast(device_type=self._device_type, enabled=self._amp_mixed_precision):
                values["humanoid"], _, _ = self.values["humanoid"].act(
                    {"states": self._amp_state_preprocessor(states["humanoid"])}, role="value"
                )
                values["humanoid"] = self._amp_value_preprocessor(values["humanoid"], inverse=True)

            # time-limit (truncation) bootstrapping
            if self._ppo_time_limit_bootstrap:
                rewards["exo"] += self._ppo_discount_factor  * values["exo"] * truncated["exo"]

            if self._amp_time_limit_bootstrap:
                rewards["humanoid"] += self._amp_discount_factor  * values["humanoid"] * truncated["humanoid"]

            # compute next values
            with torch.autocast(device_type=self._device_type, enabled=self._amp_mixed_precision):
                print("next_states",next_states)
                next_values, _, _ = self.values["humanoid"].act({"states": self._amp_state_preprocessor(next_states["humanoid"])}, role="value")
                next_values= self._amp_value_preprocessor(next_states["humanoid"], inverse=True)
                
                for key in infos:
                    print("infos key:", key)

                if "terminate" in infos:
                    next_values*= infos["terminate"].view(-1, 1).logical_not()  # compatibility with IsaacGymEnvs
                else:
                    next_values *= terminated["humanoid"].view(-1, 1).logical_not()

            for uid in self.possible_agents:
            # storage transition in memory
                self.memories[uid].add_samples(
                    states=states[uid],
                    actions=actions[uid],
                    rewards=rewards[uid],
                    next_states=next_states[uid],
                    terminated=terminated[uid],
                    truncated=truncated[uid],
                    log_prob=self._current_log_prob[uid],
                    values=values[uid],
                )

            self.memories["humanoid"].add_samples(
                amp_states=amp_states,
                next_values=next_values,
            )

            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states["humanoid"],
                    actions=actions["humanoid"],
                    rewards=rewards["humanoid"],
                    next_states=next_states["humanoid"],
                    terminated=terminated["humanoid"],
                    truncated=truncated["humanoid"],
                    log_prob=self._current_log_prob["humanoid"],
                    values=values["humanoid"],
                    amp_states=amp_states,
                    next_values=next_values,
                )
            print(f"target_values")

        


#     def pre_interaction(self, timestep: int, timesteps: int) -> None:
#         """Callback called before the interaction with the environment

#         :param timestep: Current timestep
#         :type timestep: int
#         :param timesteps: Number of timesteps
#         :type timesteps: int
#         """
#         if self.collect_observation is not None:
#             self._current_states = self.collect_observation()

#     def post_interaction(self, timestep: int, timesteps: int) -> None:
#         """Callback called after the interaction with the environment

#         :param timestep: Current timestep
#         :type timestep: int
#         :param timesteps: Number of timesteps
#         :type timesteps: int
#         """
#         self._rollout += 1
#         if not self._rollout % self._rollouts and timestep >= self._learning_starts:
#             self.set_mode("train")
#             self._update(timestep, timesteps)
#             self.set_mode("eval")

#         # write tracking data and checkpoints
#         super().post_interaction(timestep, timesteps)

#     def _update(self, timestep: int, timesteps: int) -> None:
#         """Algorithm's main update step

#         :param timestep: Current timestep
#         :type timestep: int
#         :param timesteps: Number of timesteps
#         :type timesteps: int
#         """

#         def compute_gae(
#             rewards: torch.Tensor,
#             dones: torch.Tensor,
#             values: torch.Tensor,
#             next_values: torch.Tensor,
#             discount_factor: float = 0.99,
#             lambda_coefficient: float = 0.95,
#         ) -> torch.Tensor:
#             """Compute the Generalized Advantage Estimator (GAE)

#             :param rewards: Rewards obtained by the agent
#             :type rewards: torch.Tensor
#             :param dones: Signals to indicate that episodes have ended
#             :type dones: torch.Tensor
#             :param values: Values obtained by the agent
#             :type values: torch.Tensor
#             :param next_values: Next values obtained by the agent
#             :type next_values: torch.Tensor
#             :param discount_factor: Discount factor
#             :type discount_factor: float
#             :param lambda_coefficient: Lambda coefficient
#             :type lambda_coefficient: float

#             :return: Generalized Advantage Estimator
#             :rtype: torch.Tensor
#             """
#             advantage = 0
#             advantages = torch.zeros_like(rewards)
#             not_dones = dones.logical_not()
#             memory_size = rewards.shape[0]

#             # advantages computation
#             for i in reversed(range(memory_size)):
#                 advantage = (
#                     rewards[i]
#                     - values[i]
#                     + discount_factor * (next_values[i] + lambda_coefficient * not_dones[i] * advantage)
#                 )
#                 advantages[i] = advantage
#             # returns computation
#             returns = advantages + values
#             # normalize advantages
#             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

#             return returns, advantages

#         # update dataset of reference motions
#         self.motion_dataset.add_samples(states=self.collect_reference_motions(self._amp_batch_size))

#         # compute combined rewards
#         rewards = self.memory.get_tensor_by_name("rewards")
#         amp_states = self.memory.get_tensor_by_name("amp_states")

#         with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
#             amp_logits, _, _ = self.discriminator.act(
#                 {"states": self._amp_state_preprocessor(amp_states)}, role="discriminator"
#             )
#             style_reward = -torch.log(
#                 torch.maximum(1 - 1 / (1 + torch.exp(-amp_logits)), torch.tensor(0.0001, device=self.device))
#             )
#             style_reward *= self._discriminator_reward_scale
#             style_reward = style_reward.view(rewards.shape)

#         combined_rewards = self._task_reward_weight * rewards + self._style_reward_weight * style_reward

#         # compute returns and advantages
#         values = self.memory.get_tensor_by_name("values")
#         next_values = self.memory.get_tensor_by_name("next_values")
#         returns, advantages = compute_gae(
#             rewards=combined_rewards,
#             dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
#             values=values,
#             next_values=next_values,
#             discount_factor=self._discount_factor,
#             lambda_coefficient=self._lambda,
#         )

#         self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
#         self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
#         self.memory.set_tensor_by_name("advantages", advantages)

#         # sample mini-batches from memory
#         sampled_batches = self.memory.sample_all(names=self.tensors_names, mini_batches=self._mini_batches)
#         sampled_motion_batches = self.motion_dataset.sample(
#             names=["states"], batch_size=self.memory.memory_size * self.memory.num_envs, mini_batches=self._mini_batches
#         )
#         if len(self.reply_buffer):
#             sampled_replay_batches = self.reply_buffer.sample(
#                 names=["states"],
#                 batch_size=self.memory.memory_size * self.memory.num_envs,
#                 mini_batches=self._mini_batches,
#             )
#         else:
#             sampled_replay_batches = [[batches[self.tensors_names.index("amp_states")]] for batches in sampled_batches]

#         cumulative_policy_loss = 0
#         cumulative_entropy_loss = 0
#         cumulative_value_loss = 0
#         cumulative_discriminator_loss = 0

#         # learning epochs
#         for epoch in range(self._learning_epochs):
#             kl_divergences = []

#             # mini-batches loop
#             for batch_index, (
#                 sampled_states,
#                 sampled_actions,
#                 _,
#                 _,
#                 _,
#                 sampled_log_prob,
#                 sampled_values,
#                 sampled_returns,
#                 sampled_advantages,
#                 sampled_amp_states,
#                 _,
#             ) in enumerate(sampled_batches):

#                 with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):

#                     sampled_states = self._state_preprocessor(sampled_states, train=True)

#                     _, next_log_prob, _ = self.policy.act(
#                         {"states": sampled_states, "taken_actions": sampled_actions}, role="policy"
#                     )

#                     # compute approximate KL divergence
#                     with torch.no_grad():
#                         ratio = next_log_prob - sampled_log_prob
#                         kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
#                         kl_divergences.append(kl_divergence)

#                     # compute entropy loss
#                     if self._entropy_loss_scale:
#                         entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
#                     else:
#                         entropy_loss = 0

#                     # compute policy loss
#                     ratio = torch.exp(next_log_prob - sampled_log_prob)
#                     surrogate = sampled_advantages * ratio
#                     surrogate_clipped = sampled_advantages * torch.clip(
#                         ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip
#                     )

#                     policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

#                     # compute value loss
#                     predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")

#                     if self._clip_predicted_values:
#                         predicted_values = sampled_values + torch.clip(
#                             predicted_values - sampled_values, min=-self._value_clip, max=self._value_clip
#                         )
#                     value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

#                     # compute discriminator loss
#                     if self._discriminator_batch_size:
#                         sampled_amp_states = self._amp_state_preprocessor(
#                             sampled_amp_states[0 : self._discriminator_batch_size], train=True
#                         )
#                         sampled_amp_replay_states = self._amp_state_preprocessor(
#                             sampled_replay_batches[batch_index][0][0 : self._discriminator_batch_size], train=True
#                         )
#                         sampled_amp_motion_states = self._amp_state_preprocessor(
#                             sampled_motion_batches[batch_index][0][0 : self._discriminator_batch_size], train=True
#                         )
#                     else:
#                         sampled_amp_states = self._amp_state_preprocessor(sampled_amp_states, train=True)
#                         sampled_amp_replay_states = self._amp_state_preprocessor(
#                             sampled_replay_batches[batch_index][0], train=True
#                         )
#                         sampled_amp_motion_states = self._amp_state_preprocessor(
#                             sampled_motion_batches[batch_index][0], train=True
#                         )

#                     sampled_amp_motion_states.requires_grad_(True)
#                     amp_logits, _, _ = self.discriminator.act({"states": sampled_amp_states}, role="discriminator")
#                     amp_replay_logits, _, _ = self.discriminator.act(
#                         {"states": sampled_amp_replay_states}, role="discriminator"
#                     )
#                     amp_motion_logits, _, _ = self.discriminator.act(
#                         {"states": sampled_amp_motion_states}, role="discriminator"
#                     )

#                     amp_cat_logits = torch.cat([amp_logits, amp_replay_logits], dim=0)

#                     # discriminator prediction loss
#                     discriminator_loss = 0.5 * (
#                         nn.BCEWithLogitsLoss()(amp_cat_logits, torch.zeros_like(amp_cat_logits))
#                         + torch.nn.BCEWithLogitsLoss()(amp_motion_logits, torch.ones_like(amp_motion_logits))
#                     )

#                     # discriminator logit regularization
#                     if self._discriminator_logit_regularization_scale:
#                         logit_weights = torch.flatten(list(self.discriminator.modules())[-1].weight)
#                         discriminator_loss += self._discriminator_logit_regularization_scale * torch.sum(
#                             torch.square(logit_weights)
#                         )

#                     # discriminator gradient penalty
#                     if self._discriminator_gradient_penalty_scale:
#                         amp_motion_gradient = torch.autograd.grad(
#                             amp_motion_logits,
#                             sampled_amp_motion_states,
#                             grad_outputs=torch.ones_like(amp_motion_logits),
#                             create_graph=True,
#                             retain_graph=True,
#                             only_inputs=True,
#                         )
#                         gradient_penalty = torch.sum(torch.square(amp_motion_gradient[0]), dim=-1).mean()
#                         discriminator_loss += self._discriminator_gradient_penalty_scale * gradient_penalty

#                     # discriminator weight decay
#                     if self._discriminator_weight_decay_scale:
#                         weights = [
#                             torch.flatten(module.weight)
#                             for module in self.discriminator.modules()
#                             if isinstance(module, torch.nn.Linear)
#                         ]
#                         weight_decay = torch.sum(torch.square(torch.cat(weights, dim=-1)))
#                         discriminator_loss += self._discriminator_weight_decay_scale * weight_decay

#                     discriminator_loss *= self._discriminator_loss_scale

#                 # optimization step
#                 self.optimizer.zero_grad()
#                 self.scaler.scale(policy_loss + entropy_loss + value_loss + discriminator_loss).backward()

#                 if config.torch.is_distributed:
#                     self.policy.reduce_parameters()
#                     self.value.reduce_parameters()
#                     self.discriminator.reduce_parameters()

#                 if self._grad_norm_clip > 0:
#                     self.scaler.unscale_(self.optimizer)
#                     nn.utils.clip_grad_norm_(
#                         itertools.chain(
#                             self.policy.parameters(), self.value.parameters(), self.discriminator.parameters()
#                         ),
#                         self._grad_norm_clip,
#                     )

#                 self.scaler.step(self.optimizer)
#                 self.scaler.update()

#                 # update cumulative losses
#                 cumulative_policy_loss += policy_loss.item()
#                 cumulative_value_loss += value_loss.item()
#                 if self._entropy_loss_scale:
#                     cumulative_entropy_loss += entropy_loss.item()
#                 cumulative_discriminator_loss += discriminator_loss.item()

#             # update learning rate
#             if self._learning_rate_scheduler:
#                 if isinstance(self.scheduler, KLAdaptiveLR):
#                     kl = torch.tensor(kl_divergences, device=self.device).mean()
#                     # reduce (collect from all workers/processes) KL in distributed runs
#                     if config.torch.is_distributed:
#                         torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
#                         kl /= config.torch.world_size
#                     self.scheduler.step(kl.item())
#                 else:
#                     self.scheduler.step()

#         # update AMP replay buffer
#         self.reply_buffer.add_samples(states=amp_states.view(-1, amp_states.shape[-1]))

#         # record data
#         self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
#         self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
#         if self._entropy_loss_scale:
#             self.track_data(
#                 "Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches)
#             )
#         self.track_data(
#             "Loss / Discriminator loss", cumulative_discriminator_loss / (self._learning_epochs * self._mini_batches)
#         )

#         self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

#         if self._learning_rate_scheduler:
#             self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
